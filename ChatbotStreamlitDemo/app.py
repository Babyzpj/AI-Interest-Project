#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：LLMStreamlit
# @IDE     ：PyCharm
# @Author  ：Huang Andy Hong Hua
# @Email   ：
# @Date    ：2024/3/20 10:33
# ====================================
import streamlit as st
import os, json
import embed_pdf
from utils.sagemaker_endpoint import SagemakerEndpointEmbeddings
from handlers.content import ContentHandler, ContentHandlerQA
from handlers.custom_aws_endpoint import SagemakerStreamContentHandler, CustomSagemakerLLMEndpoint
from langchain.callbacks.streamlit.streamlit_callback_handler import StreamlitCallbackHandler
from langchain_community.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.llms import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from utils.debug_tools import print_log, print_prompt

# ==================   init =====================
chatbot_config = json.load(open('./configs/config.json'))
REGION = 'cn-northwest-1'
EMBEDDING_ENDPOINT_NAME = "cmlm-bge-g4dn-endpoint"
LLM_ENDPOINT_NAME = chatbot_config["chatbot"]["llm_endpoint_name"]
STREAM = True
# STREAM =False
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
# print(own_embeddings.embed_query("test"))

# create sidebar and ask for openai api key if not set in secrets
secrets_file_path = os.path.join(".streamlit", "secrets.toml")
if os.path.exists(secrets_file_path):
    try:
        if "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        else:
            print("OpenAI API Key not found in environment variables")
    except FileNotFoundError:
        print('Secrets file not found')
else:
    print('Secrets file not found')

if not os.getenv('OPENAI_API_KEY', '').startswith("sk-"):
    os.environ["OPENAI_API_KEY"] = st.sidebar.text_input(
        "OpenAI API Key", type="password"
    )
else:
    if st.sidebar.button("Embed Documents"):
        st.sidebar.info("Embedding documents...")
        try:
            embed_pdf.embed_all_pdf_docs(own_embeddings)
            st.sidebar.info("Done!")
        except Exception as e:
            st.sidebar.error(e)
            st.sidebar.error("Failed to embed documents.")

# create the app
st.title("Welcome to NimaGPT")

chosen_file = st.radio(
    "Choose a file to search", embed_pdf.get_all_index_files(), index=0
)

# check if openai api key is set
if not os.getenv('OPENAI_API_KEY', '').startswith("sk-"):
    st.warning("Please enter your OpenAI API key!", icon="⚠")
    st.stop()

# load the agent
from llm_helper import convert_message, get_rag_chain, \
    get_rag_fusion_chain, get_react_agent_chain, get_self_ask_agent_chain

rag_method_map = {
    'Basic RAG': get_rag_chain,
    'RAG Fusion': get_rag_fusion_chain,
    'RAG react Agent': get_react_agent_chain,
    'RAG self-ask-with-search Agent': get_self_ask_agent_chain,
}
chosen_rag_method = st.radio(
    "Choose a RAG method", rag_method_map.keys(), index=0
)
get_rag_chain_func = rag_method_map[chosen_rag_method]
## get the chain WITHOUT the retrieval callback (not used)
# custom_chain = get_rag_chain_func(chosen_file)

# create the message history state
if "messages" not in st.session_state:
    st.session_state.messages = []

# render older messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# render the chat input
prompt = st.chat_input("Enter your message...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    # render the user's new message
    with st.chat_message("user"):
        st.markdown(prompt)

    # render the assistant's response
    with st.chat_message("assistant"):
        retrival_container = st.container()
        message_placeholder = st.empty()

        retrieval_status = retrival_container.status("**Context Retrieval**")
        queried_questions = []
        rendered_questions = set()


        def update_retrieval_status():
            for q in queried_questions:
                if q in rendered_questions:
                    continue
                rendered_questions.add(q)
                retrieval_status.markdown(f"\n\n`- {q}`")


        def retrieval_cb(qs):
            for q in qs:
                if q not in queried_questions:
                    queried_questions.append(q)
            return qs


        # get the chain with the retrieval callback
        st_cb = StreamlitCallbackHandler(st.container())
        custom_chain = get_rag_chain_func(chosen_file, retrieval_cb=retrieval_cb, st_cb=st_cb)

        if "messages" in st.session_state:
            chat_history = [convert_message(m) for m in st.session_state.messages[:-1]]
        else:
            chat_history = []

        full_response = ""
        # *****************    Agent 还是非 Agent *************************
        if "Agent" in chosen_rag_method:  # Agent
            print(chosen_rag_method)
            response = custom_chain.invoke({"input": prompt, "chat_history": chat_history})
            if type(response) == str:
                full_response += response
            elif "output" in response:
                full_response += response["output"]
            else:
                full_response += response.content

            message_placeholder.markdown(full_response + "▌")
            update_retrieval_status()
        else:  # RAG 使用
            print(chosen_rag_method)
            for response in custom_chain.stream(
                    {"input": prompt, "chat_history": chat_history}
            ):
                if type(response) == str:
                    full_response += response
                elif "output" in response:
                    full_response += response["output"]
                else:
                    full_response += response.content

                message_placeholder.markdown(full_response + "▌")
                update_retrieval_status()
        # *****************    Agent 还是非 Agent *************************
        retrieval_status.update(state="complete")
        message_placeholder.markdown(full_response)

    # add the full response to the message history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
