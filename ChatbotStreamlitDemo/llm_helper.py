#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：LLMStreamlit
# @IDE     ：PyCharm
# @Author  ：Huang Andy Hong Hua
# @Email   ：
# @Date    ：2024/3/20 10:33
# ====================================
from typing import Optional

# langchain imports
from langchain import hub
from langchain_community.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.schema.runnable import RunnableMap
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.agents import initialize_agent, Tool, \
    create_openai_tools_agent, create_react_agent, create_self_ask_with_search_agent
from operator import itemgetter
# from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks.streamlit.streamlit_callback_handler import StreamlitCallbackHandler
import json
from utils.sagemaker_endpoint import SagemakerEndpointEmbeddings
from handlers.content import ContentHandler, ContentHandlerQA
from langchain_community.llms import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from handlers.custom_aws_endpoint import SagemakerStreamContentHandler, CustomSagemakerLLMEndpoint
from utils.debug_tools import print_log, print_prompt
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import tool, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from agent_helper import retry_and_streamlit_callback

# ==================   init =====================
chatbot_config = json.load(open('./configs/config.json'))
REGION = 'cn-northwest-1'
EMBEDDING_ENDPOINT_NAME = "cmlm-bge-g4dn-endpoint"
LLM_ENDPOINT_NAME = chatbot_config["chatbot"]["llm_endpoint_name"]
STREAM = True
# STREAM = False
RESET = '/rs'
STOP = [f"\nuser", ]
content_handler = ContentHandler()
content_handler_qa = ContentHandlerQA()
# ==================   init =====================

# =============== embedding ====================
own_embeddings = SagemakerEndpointEmbeddings(
    endpoint_name=EMBEDDING_ENDPOINT_NAME,
    region_name=REGION,
    content_handler=content_handler,
)
# =============== embedding ====================


# ***************** llm *****************
own_llm = ChatOpenAI(api_key=chatbot_config["chatbot"]["moonshot_api_key"],
                     base_url=chatbot_config["chatbot"]["moonshot_api_base"],
                     model=chatbot_config["chatbot"]["moonshot_deployment_name"],
                     verbose=True)

# ***************** llm *****************


# messages = [
#     SystemMessage(content="You are a helpful assistant."),
#     HumanMessage(content="hi")
# ]
# print(own_llm(messages))


def format_docs(docs):
    res = ""
    # res = str(docs)
    for doc in docs:
        escaped_page_content = doc.page_content.replace("\n", "\\n")
        res += "<doc>\n"
        res += f"  <content>{escaped_page_content}</content>\n"
        for m in doc.metadata:
            res += f"  <{m}>{doc.metadata[m]}</{m}>\n"
        res += "</doc>\n"
    return res


def get_search_index(file_name="test.pdf", index_folder="index"):
    global own_embeddings
    # load embeddings
    from langchain.vectorstores import FAISS
    from langchain.embeddings.openai import OpenAIEmbeddings

    search_index = FAISS.load_local(
        folder_path=index_folder,
        index_name=file_name + ".index",
        embeddings=own_embeddings,
    )
    return search_index


def convert_message(m):
    if m["role"] == "user":
        return HumanMessage(content=m["content"])
    elif m["role"] == "assistant":
        return AIMessage(content=m["content"])
    elif m["role"] == "system":
        return SystemMessage(content=m["content"])
    else:
        raise ValueError(f"Unknown role {m['role']}")


_condense_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {input}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_condense_template)

_rag_template = """Answer the question based only on the following context, citing the page number(s) of the document(s) you used to answer the question:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(_rag_template)


def _format_chat_history(chat_history):
    def format_single_chat_message(m):
        if type(m) is HumanMessage:
            return "Human: " + m.content
        elif type(m) is AIMessage:
            return "Assistant: " + m.content
        elif type(m) is SystemMessage:
            return "System: " + m.content
        else:
            raise ValueError(f"Unknown role {m['role']}")

    return "\n".join([format_single_chat_message(m) for m in chat_history])


def get_standalone_question_from_chat_history_chain():
    global own_llm
    _inputs = RunnableMap(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
                            | CONDENSE_QUESTION_PROMPT
                            | own_llm  # ChatOpenAI(temperature=0)
                            | StrOutputParser(),
    )
    return _inputs


def get_rag_chain(file_name="test.pdf", index_folder="index", retrieval_cb=None, **kwargs):
    global own_llm
    vectorstore = get_search_index(file_name, index_folder)
    retriever = vectorstore.as_retriever()

    if retrieval_cb is None:
        retrieval_cb = lambda x: x

    def context_update_fn(q):
        retrieval_cb([q])
        return q

    _inputs = RunnableMap(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
                            | CONDENSE_QUESTION_PROMPT
                            | own_llm  # ChatOpenAI(temperature=0)
                            | StrOutputParser(),
    )
    _context = {
        "context": itemgetter("standalone_question") | RunnablePassthrough(context_update_fn) | retriever | format_docs,
        "question": lambda x: x["standalone_question"],
    }
    conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | own_llm  # ChatOpenAI()   # | print_prompt
    return conversational_qa_chain


# RAG fusion chain
# source1: https://youtu.be/GchC5WxeXGc?si=6i7J0rPZI7SNwFYZ
# source2: https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1
def reciprocal_rank_fusion(results: list[list], k=60):
    from langchain.load import dumps, loads
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results


def get_search_query_generation_chain():
    global own_llm
    from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
    prompt = ChatPromptTemplate(
        input_variables=['original_query'],
        messages=[
            SystemMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=[],
                    template='You are a helpful assistant that generates multiple search queries based on a single input query.'
                )
            ),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=['original_query'],
                    template='Generate multiple search queries related to: {original_query} \n OUTPUT (4 queries):'
                )
            )
        ]
    )

    generate_queries = (
            prompt |
            own_llm |  # ChatOpenAI(temperature=0)
            StrOutputParser() |
            (lambda x: x.split("\n"))
    )

    return generate_queries


def get_rag_fusion_chain(file_name="test.pdf", index_folder="index", retrieval_cb=None):
    global own_llm
    vectorstore = get_search_index(file_name, index_folder)
    retriever = vectorstore.as_retriever()
    query_generation_chain = get_search_query_generation_chain()
    _inputs = RunnableMap(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
                            | CONDENSE_QUESTION_PROMPT
                            | own_llm  # ChatOpenAI(temperature=0) 使用自己的模型
                            | StrOutputParser(),
    )

    if retrieval_cb is None:
        retrieval_cb = lambda x: x

    _context = {
        "context":
            RunnablePassthrough.assign(
                original_query=lambda x: x["standalone_question"]
            )
            | query_generation_chain
            | retrieval_cb
            | retriever.map()
            | reciprocal_rank_fusion
            | (lambda x: [item[0] for item in x])
            | format_docs,
        "question": lambda x: x["standalone_question"],
    }

    # conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | ChatOpenAI()
    conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | print_prompt | own_llm
    return conversational_qa_chain


#######################################################################
# Adding agent chain with OpenAI function calling

def get_search_tool_from_index(search_index, st_cb: Optional[StreamlitCallbackHandler] = None, ):
    @tool()
    @retry_and_streamlit_callback(st_cb=st_cb, tool_name="Content Seach Tool")
    def search(query: str) -> str:
        """
        Search the contents of the source document for the queries.
        :param query: the queries
        :return: Search the contents of the source
        """

        docs = search_index.similarity_search(query, k=5)
        # print('=========== use search tool,query:', query)
        # print(format_docs(docs))
        res = AgentFinish(
            return_values={"output": format_docs(docs).strip()},
            log=format_docs(docs),
        )
        return res

    return search


def get_intermediate_answer_tool_from_index(custom_agent_executor, st_cb: Optional[StreamlitCallbackHandler] = None):
    """
    获取 中间回答的工具
    :param custom_agent_executor: 自定义的agent_executor ，可以是 react
    :param st_cb:
    :return:
    """

    @tool()
    @retry_and_streamlit_callback(st_cb=st_cb, tool_name="Intermediate Answer")
    def tavily_answer(query: str) -> str:
        """return intermediate answer for the queries."""
        answer = custom_agent_executor.invoke({
            "input": query,
            "chat_history": [],
        })
        res = AgentFinish(
            return_values={"output": answer},
            log=answer,
        )
        return res

    tavily_answer.name = "Intermediate Answer"
    return tavily_answer


def get_search_tool_to_agent_use(file_name: str = "test.pdf", index_folder: str = "index",
                                 st_cb: Optional[StreamlitCallbackHandler] = None, ):
    from langchain.tools.render import format_tool_to_openai_tool
    search_index = get_search_index(file_name, index_folder)
    search_tool = Tool.from_function(
        func=get_search_tool_from_index(search_index=search_index, st_cb=st_cb),
        name="DeepSearch",
        description="DeepSearch tool provides quick and accurate knowledge base results from text queries."
    )
    lc_tools = [search_tool]
    oai_tools = [format_tool_to_openai_tool(t) for t in lc_tools]
    return lc_tools, oai_tools


def get_tavily_answer_tool_to_agent_use(file_name: str = "test.pdf", index_folder: str = "index",
                                        st_cb: Optional[StreamlitCallbackHandler] = None, ):
    search_index = get_search_index(file_name, index_folder)
    custom_agent_executor = get_react_agent_chain(file_name, index_folder, st_cb=st_cb)  # react
    tools = [
        get_intermediate_answer_tool_from_index(
            custom_agent_executor=custom_agent_executor,
            st_cb=st_cb)
    ]

    return tools


def get_lc_oai_tools(file_name: str = "test.pdf", index_folder: str = "index",
                     st_cb: Optional[StreamlitCallbackHandler] = None, ):
    from langchain.tools.render import format_tool_to_openai_tool
    search_index = get_search_index(file_name, index_folder)
    lc_tools = [get_search_tool_from_index(search_index=search_index, st_cb=st_cb)]
    oai_tools = [format_tool_to_openai_tool(t) for t in lc_tools]
    return lc_tools, oai_tools


def get_rag_react_agent_chain(file_name="test.pdf", index_folder="index", callbacks=None,
                              st_cb: Optional[StreamlitCallbackHandler] = None, retrieval_cb=None, **kwargs):
    global own_llm
    if callbacks is None:
        callbacks = []
    lc_tools, oai_tools = get_lc_oai_tools(file_name, index_folder, st_cb)
    agent = (
            get_rag_chain(file_name, index_folder, retrieval_cb)
            | OpenAIToolsAgentOutputParser()  # init_prompt
    )
    agent_executor = AgentExecutor(
        agent=agent,
        tools=lc_tools,
        # allowed_tools=tool_names,
        verbose=True,
        callbacks=callbacks,
        handle_parsing_errors=True,
    )
    return agent_executor


def get_react_agent_chain(file_name="test.pdf", index_folder="index", callbacks=None,
                          st_cb: Optional[StreamlitCallbackHandler] = None, retrieval_cb=None, **kwargs):
    global own_llm
    if callbacks is None:
        callbacks = []

    lc_tools, oai_tools = get_lc_oai_tools(file_name, index_folder, st_cb)
    # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")
    llm = own_llm  # 使用自己的模型

    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/react-chat")
    agent = create_react_agent(llm, lc_tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=lc_tools,
        # allowed_tools=tool_names,
        verbose=True,
        callbacks=callbacks,
        handle_parsing_errors=True,
    )

    # ************* initialize_agent *******************
    #  REACT_DOCSTORE , SELF_ASK_WITH_SEARCH
    # agent_executor = initialize_agent(lc_tools, llm, agent=AgentType.REACT_DOCSTORE, handle_parsing_errors=True,
    #                                   verbose=True)
    # agent_executor.agent.prompt = prompt
    # ************* initialize_agent *******************

    return agent_executor


def get_self_ask_agent_chain(file_name="test.pdf", index_folder="index", callbacks=None,
                             st_cb: Optional[StreamlitCallbackHandler] = None, retrieval_cb=None, **kwargs):
    global own_llm
    if callbacks is None:
        callbacks = []

    tools = get_tavily_answer_tool_to_agent_use(file_name, index_folder, st_cb)
    # tools[0].name = "Intermediate Answer"  # 工具名 需要固定赋值
    tool_names = [tool.name for tool in tools]

    # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")
    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/self-ask-with-search")
    agent = create_self_ask_with_search_agent(own_llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        allowed_tools=tool_names,
        verbose=True,
        callbacks=callbacks,
        handle_parsing_errors=True,
    )
    return agent_executor


if __name__ == "__main__":
    question_generation_chain = get_search_query_generation_chain()
    print('=' * 50)
    print('RAG Chain')
    chain = get_rag_chain()
    print(chain.invoke({'input': "帮我找问题‘Rehire员工是否适用ERP’在第几页", 'chat_history': []}))
    #
    # print('=' * 50)
    # print('Question Generation Chain')
    # print(question_generation_chain.invoke({'original_query': 'serverless computing'}))
    #
    # print('-' * 50)
    # print('RAG Fusion Chain')
    # chain = get_rag_fusion_chain()
    # print(chain.invoke({'input': 'serverless computing', 'chat_history': []}))

    # agent_executor = get_rag_react_agent_chain()
    # # agent_executor = get_react_agent_chain()
    # # agent_executor = get_self_ask_agent_chain()
    # # query = "ERP成功推荐的奖励机制是怎样的？"
    # query = "帮我找问题‘Rehire员工是否适用ERP’在第几页"
    # print(
    #     agent_executor.invoke({
    #         "input": query,
    #         "chat_history": [],
    #     })
    # )

    # for chunk in agent_executor.stream({"input": "Rehire员工是否适用ERP在第几页", "chat_history": []}):
    #     # Agent Action
    #     if "actions" in chunk:
    #         for action in chunk["actions"]:
    #             print(
    #                 f"Calling Tool ```{action.tool}``` with input ```{action.tool_input}```"
    #             )
    #     # Observation
    #     elif "steps" in chunk:
    #         for step in chunk["steps"]:
    #             print(f"Got result: ```{step.observation}```")
