"""Sagemaker InvokeEndpoint API."""
import io
import json
from abc import abstractmethod
from typing import Any, Dict, Generic, Iterator, List, Mapping, Optional, TypeVar, Union, AsyncIterator

from langchain_core.language_models.llms import LLM
from langchain_community.llms.utils import enforce_stop_tokens
import json
from langchain.pydantic_v1 import Extra, root_validator
from langchain.callbacks.base import BaseCallbackHandler
import io
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import run_in_executor

INPUT_TYPE = TypeVar("INPUT_TYPE", bound=Union[str, List[str]])
OUTPUT_TYPE = TypeVar("OUTPUT_TYPE", bound=Union[str, List[List[float]], Iterator])


class LineIterator:
    """
    A helper class for parsing the byte stream input.

    The output of the model will be in the following format:

    b'{"outputs": [" a"]}\n'
    b'{"outputs": [" challenging"]}\n'
    b'{"outputs": [" problem"]}\n'
    ...

    While usually each PayloadPart event from the event stream will
    contain a byte array with a full json, this is not guaranteed
    and some of the json objects may be split acrossPayloadPart events.

    For example:

    {'PayloadPart': {'Bytes': b'{"outputs": '}}
    {'PayloadPart': {'Bytes': b'[" problem"]}\n'}}


    This class accounts for this by concatenating bytes written via the 'write' function
    and then exposing a method which will return lines (ending with a '\n' character)
    within the buffer via the 'scan_lines' function.
    It maintains the position of the last read position to ensure
    that previous bytes are not exposed again.

    For more details see:
    https://aws.amazon.com/blogs/machine-learning/elevating-the-generative-ai-experience-introducing-streaming-support-in-amazon-sagemaker-hosting/
    """

    def __init__(self, stream: Any) -> None:
        self.byte_iterator = iter(stream)
        self.buffer = io.BytesIO()
        self.read_pos = 0

    def __iter__(self) -> "LineIterator":
        return self

    def __next__(self) -> Any:
        while True:
            self.buffer.seek(self.read_pos)
            line = self.buffer.readline()
            if line and line[-1] == ord("\n"):
                self.read_pos += len(line)
                return line[:-1]
            try:
                chunk = next(self.byte_iterator)
            except StopIteration:
                if self.read_pos < self.buffer.getbuffer().nbytes:
                    continue
                raise
            if "PayloadPart" not in chunk:
                # Unknown Event Type
                continue
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk["PayloadPart"]["Bytes"])


class ContentHandlerBase(Generic[INPUT_TYPE, OUTPUT_TYPE]):
    """A handler class to transform input from LLM to a
    format that SageMaker endpoint expects.

    Similarly, the class handles transforming output from the
    SageMaker endpoint to a format that LLM class expects.
    """

    """
    Example:
        .. code-block:: python

            class ContentHandler(ContentHandlerBase):
                content_type = "application/json"
                accepts = "application/json"

                def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
                    input_str = json.dumps({prompt: prompt, **model_kwargs})
                    return input_str.encode('utf-8')

                def transform_output(self, output: bytes) -> str:
                    response_json = json.loads(output.read().decode("utf-8"))
                    return response_json[0]["generated_text"]
    """

    content_type: Optional[str] = "text/plain"
    """The MIME type of the input data passed to endpoint"""

    accepts: Optional[str] = "text/plain"
    """The MIME type of the response data returned from endpoint"""

    @abstractmethod
    def transform_input(self, prompt: INPUT_TYPE, model_kwargs: Dict) -> bytes:
        """Transforms the input to a format that model can accept
        as the request Body. Should return bytes or seekable file
        like object in the format specified in the content_type
        request header.
        """

    @abstractmethod
    def transform_output(self, output: bytes) -> OUTPUT_TYPE:
        """Transforms the output from the model to string that
        the LLM class expects.
        """


class LLMContentHandler(ContentHandlerBase[str, str]):
    """Content handler for LLM class."""


class StreamScanner:
    def __init__(self):
        self.buff = io.BytesIO()
        self.read_pos = 0

    def write(self, content):
        self.buff.seek(0, io.SEEK_END)
        self.buff.write(content)

    def readlines(self):
        self.buff.seek(self.read_pos)
        for line in self.buff.readlines():
            if line[-1] != b'\n':
                self.read_pos += len(line)
                yield line[:-1]

    def reset(self):
        self.read_pos = 0


class SagemakerStreamContentHandler(LLMContentHandler):
    content_type: Optional[str] = "application/json"
    accepts: Optional[str] = "application/json"
    callbacks: BaseCallbackHandler

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    def __init__(self, callbacks: BaseCallbackHandler, stop, stream, frequency: int = 5, **kwargs) -> None:

        super().__init__(**kwargs)
        self.callbacks = callbacks
        self.stop = stop
        self.frequency = frequency
        self.stream = stream

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        print(prompt, model_kwargs)
        input_str = json.dumps({'inputs': prompt, **model_kwargs})
        return input_str.encode('utf-8')

    def transform_output(self, event_stream: Any) -> str:
        if self.stream:
            scanner = StreamScanner()
            text = ''
            count = 0
            for event in event_stream:
                scanner.write(event)
                for line in scanner.readlines():
                    try:
                        resp = json.loads(line)
                        token = resp.get("outputs")['outputs']
                        text += token
                        for stop in self.stop:  ##如果碰到STOP截断
                            if text.endswith(stop):
                                if self.callbacks:
                                    self.callbacks.on_llm_end(None)
                                text = text.rstrip(stop)
                                return text
                        # self.callbacks.on_llm_new_token(token)
                        count += 1
                        if count % self.frequency == 0:
                            if self.callbacks:
                                self.callbacks.on_llm_new_token(text)
                        # print(token, end='')
                    except Exception as e:
                        # print(line)
                        continue
            if self.callbacks:
                self.callbacks.on_llm_end(None)
            return text
        else:
            response_json = json.loads(event_stream.read().decode("utf-8"))
            return response_json["answers"]


class CustomSagemakerLLMEndpoint(LLM):
    """Sagemaker Inference Endpoint models.

    To use, you must supply the endpoint name from your deployed
    Sagemaker model & the region where it is deployed.

    To authenticate, the AWS client uses the following methods to
    automatically load credentials:
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If a specific credential profile should be used, you must pass
    the name of the profile from the ~/.aws/credentials file that is to be used.

    Make sure the credentials / roles used have the required policies to
    access the Sagemaker endpoint.
    See: https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies.html
    """

    """
    Args:        

        region_name: The aws region e.g., `us-west-2`.
            Fallsback to AWS_DEFAULT_REGION env variable
            or region specified in ~/.aws/config.

        credentials_profile_name: The name of the profile in the ~/.aws/credentials
            or ~/.aws/config files, which has either access keys or role information
            specified. If not specified, the default credential profile or, if on an
            EC2 instance, credentials from IMDS will be used.

        client: boto3 client for Sagemaker Endpoint

        content_handler: Implementation for model specific LLMContentHandler 


    Example:
        .. code-block:: python

            from langchain_community.llms import SagemakerEndpoint
            endpoint_name = (
                "my-endpoint-name"
            )
            region_name = (
                "us-west-2"
            )
            credentials_profile_name = (
                "default"
            )
            se = SagemakerEndpoint(
                endpoint_name=endpoint_name,
                region_name=region_name,
                credentials_profile_name=credentials_profile_name
            )

        #Use with boto3 client
            client = boto3.client(
                        "sagemaker-runtime",
                        region_name=region_name
                    )

            se = SagemakerEndpoint(
                endpoint_name=endpoint_name,
                client=client
            )

    """
    client: Any = None
    """Boto3 client for sagemaker runtime"""

    endpoint_name: str = ""
    """The name of the endpoint from the deployed Sagemaker model.
    Must be unique within an AWS Region."""

    region_name: str = ""
    """The aws region where the Sagemaker model is deployed, eg. `us-west-2`."""

    credentials_profile_name: Optional[str] = None
    """The name of the profile in the ~/.aws/credentials or ~/.aws/config files, which
    has either access keys or role information specified.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
    """

    content_handler: LLMContentHandler
    """The content handler class that provides an input and
    output transform functions to handle formats between LLM
    and the endpoint.
    """

    streaming: bool = False
    """Whether to stream the results."""

    """
     Example:
        .. code-block:: python

        from langchain_community.llms.sagemaker_endpoint import LLMContentHandler

        class ContentHandler(LLMContentHandler):
                content_type = "application/json"
                accepts = "application/json"

                def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
                    input_str = json.dumps({prompt: prompt, **model_kwargs})
                    return input_str.encode('utf-8')

                def transform_output(self, output: bytes) -> str:
                    response_json = json.loads(output.read().decode("utf-8"))
                    return response_json[0]["generated_text"]
    """

    model_kwargs: Optional[Dict] = None
    """Keyword arguments to pass to the model."""

    endpoint_kwargs: Optional[Dict] = None
    """Optional attributes passed to the invoke_endpoint
    function. See `boto3`_. docs for more info.
    .. _boto3: <https://boto3.amazonaws.com/v1/documentation/api/latest/index.html>
    """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Dont do anything if client provided externally"""
        if values.get("client") is not None:
            return values

        """Validate that AWS credentials to and python package exists in environment."""
        try:
            import boto3

            try:
                if values["credentials_profile_name"] is not None:
                    session = boto3.Session(
                        profile_name=values["credentials_profile_name"]
                    )
                else:
                    # use default credentials
                    session = boto3.Session()

                values["client"] = session.client(
                    "sagemaker-runtime", region_name=values["region_name"]
                )

            except Exception as e:
                raise ValueError(
                    "Could not load credentials to authenticate with AWS client. "
                    "Please check that credentials in the specified "
                    "profile name are valid."
                ) from e

        except ImportError:
            raise ImportError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"endpoint_name": self.endpoint_name},
            **{"model_kwargs": _model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "sagemaker_endpoint"

    def _messages_to_prompt(self, messages):
        # Process each message object and format it as a string
        return '\n'.join([f'{message.__class__.__name__}: {message.content}' for message in messages])

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """Call out to Sagemaker inference endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = se("Tell me a joke.")
        """
        _model_kwargs = self.model_kwargs or {}
        _model_kwargs = {**_model_kwargs, **kwargs}
        _endpoint_kwargs = self.endpoint_kwargs or {}
        if type(prompt) == list:  # 如果是传 openai的messages格式，先转变一下
            prompt = self._messages_to_prompt(prompt)
        body = self.content_handler.transform_input(prompt, _model_kwargs)
        content_type = self.content_handler.content_type
        accepts = self.content_handler.accepts

        if self.streaming and run_manager:
            try:
                resp = self.client.invoke_endpoint_with_response_stream(
                    EndpointName=self.endpoint_name,
                    Body=body,
                    ContentType=self.content_handler.content_type,
                    **_endpoint_kwargs,
                )
                iterator = LineIterator(resp["Body"])
                current_completion: str = ""
                for line in iterator:
                    resp = json.loads(line)
                    resp_output = resp.get("outputs")[0]
                    if stop is not None:
                        # Uses same approach as below
                        resp_output = enforce_stop_tokens(resp_output, stop)
                    current_completion += resp_output
                    run_manager.on_llm_new_token(resp_output)
                return current_completion
            except Exception as e:
                raise ValueError(f"Error raised by streaming inference endpoint: {e}")
        else:
            try:
                response = self.client.invoke_endpoint(
                    EndpointName=self.endpoint_name,
                    Body=body,
                    ContentType=content_type,
                    Accept=accepts,
                    **_endpoint_kwargs,
                )
            except Exception as e:
                raise ValueError(f"Error raised by inference endpoint: {e}")

            text = self.content_handler.transform_output(response["Body"])
            if stop is not None:
                # This is a bit hacky, but I can't figure out a better way to enforce
                # stop tokens when making calls to the sagemaker endpoint.
                text = enforce_stop_tokens(text, stop)

            return text


class CustomChatModelSagemakerEndpoint(BaseChatModel):
    """A custom chat model that echoes the first `n` characters of the input.

    When contributing an implementation to LangChain, carefully document
    the model including the initialization parameters, include
    an example of how to initialize the model and include any relevant
    links to the underlying models documentation or API.

    Example:

        .. code-block:: python

            model = CustomChatModel(n=2)
            result = model.invoke([HumanMessage(content="hello")])
            result = model.batch([[HumanMessage(content="hello")],
                                 [HumanMessage(content="world")]])
    """

    n: int
    """The number of characters from the last message of the prompt to be echoed."""

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        """Override the _generate method to implement the chat model logic.

        This can be a call to an API, a call to a local model, or any other
        implementation that generates a response to the input prompt.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        last_message = messages[-1]
        tokens = last_message.content[: self.n]
        message = AIMessage(content=tokens)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model.

        This method should be implemented if the model can generate output
        in a streaming fashion. If the model does not support streaming,
        do not implement it. In that case streaming requests will be automatically
        handled by the _generate method.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        last_message = messages[-1]
        tokens = last_message.content[: self.n]

        for token in tokens:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))

            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)

            yield chunk

    async def _astream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """An async variant of astream.

        If not provided, the default behavior is to delegate to the _generate method.

        The implementation below instead will delegate to `_stream` and will
        kick it off in a separate thread.

        If you're able to natively support async, then by all means do so!
        """
        result = await run_in_executor(
            None,
            self._stream,
            messages,
            stop=stop,
            run_manager=run_manager.get_sync() if run_manager else None,
            **kwargs,
        )
        for chunk in result:
            yield chunk

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "echoing-chat-model-advanced"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {"n": self.n}
