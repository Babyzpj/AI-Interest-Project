import tiktoken
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.sagemaker_endpoint import SagemakerEndpointEmbeddings
from handlers.content import ContentHandler, ContentHandlerQA
from handlers.custom_aws_endpoint import SagemakerStreamContentHandler, CustomSagemakerLLMEndpoint
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain import SagemakerEndpoint
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from utils.debug_tools import print_log, print_prompt
from langchain_community.vectorstores import Chroma
import json
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import umap
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.mixture import GaussianMixture
from langchain import hub
from langchain_core.runnables import RunnablePassthrough

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
RANDOM_SEED = 224  # Fixed seed for reproducibility
# ==================   init =====================

# =============== embedding ====================
own_embeddings = SagemakerEndpointEmbeddings(
    endpoint_name=EMBEDDING_ENDPOINT_NAME,
    region_name=REGION,
    content_handler=content_handler,
)
# =============== embedding ====================


# ***************** llm *****************
# ============= moonshot ====================
# own_llm = ChatOpenAI(
#     api_key=chatbot_config["chatbot"]["moonshot_api_key"],
#     base_url=chatbot_config["chatbot"]["moonshot_api_base"],
#     model=chatbot_config["chatbot"]["moonshot_deployment_name"],
#     verbose=True)
# ============= moonshot ====================

# ============= azure ====================
own_llm = AzureChatOpenAI(
    api_key=chatbot_config["chatbot"]["azureopenai_api_key"],
    azure_endpoint=chatbot_config["chatbot"]["azureopenai_api_base"],
    azure_deployment=chatbot_config["chatbot"]["azureopenai_deployment_name"],
    openai_api_version=chatbot_config["chatbot"]["azureopenai_api_version"],
)


# ============= azure ====================

# # ============== aws sagemaker endpoint =======================
# stream_content_handler = SagemakerStreamContentHandler(
#     callbacks=[], stop=STOP, stream=STREAM  # callbacks=StreamingStdOutCallbackHandler()
# )
#
# parameters = {"top_k": 1, "top_p": 0}
# model_kwargs = {'parameters': parameters, 'history': [], 'stream': STREAM}
# own_llm = CustomSagemakerLLMEndpoint(endpoint_name=LLM_ENDPOINT_NAME, region_name=REGION,
#                                      content_handler=stream_content_handler, model_kwargs=model_kwargs)
# # ============== aws sagemaker endpoint =======================
# ***************** llm *****************

# messages = [
#     SystemMessage(content="You are a helpful assistant."),
#     HumanMessage(content="hi")
# ]
# print(own_llm(messages))
# print(own_embeddings.embed_query("test"))


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


### --- Code from citations referenced above (added comments and docstrings) --- ###


def global_cluster_embeddings(
        embeddings: np.ndarray,
        dim: int,
        n_neighbors: Optional[int] = None,
        metric: str = "cosine",
) -> np.ndarray:
    """
    Perform global dimensionality reduction on the embeddings using UMAP.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for the reduced space.
    - n_neighbors: Optional; the number of neighbors to consider for each point.
                   If not provided, it defaults to the square root of the number of embeddings.
    - metric: The distance metric to use for UMAP.

    Returns:
    - A numpy array of the embeddings reduced to the specified dimensionality.
    """
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def local_cluster_embeddings(
        embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    """
    Perform local dimensionality reduction on the embeddings using UMAP, typically after global clustering.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for the reduced space.
    - num_neighbors: The number of neighbors to consider for each point.
    - metric: The distance metric to use for UMAP.

    Returns:
    - A numpy array of the embeddings reduced to the specified dimensionality.
    """
    return umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def get_optimal_clusters(
        embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    """
    Determine the optimal number of clusters using the Bayesian Information Criterion (BIC) with a Gaussian Mixture Model.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - max_clusters: The maximum number of clusters to consider.
    - random_state: Seed for reproducibility.

    Returns:
    - An integer representing the optimal number of clusters found.
    """
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    return n_clusters[np.argmin(bics)]


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    """
    Cluster embeddings using a Gaussian Mixture Model (GMM) based on a probability threshold.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - threshold: The probability threshold for assigning an embedding to a cluster.
    - random_state: Seed for reproducibility.

    Returns:
    - A tuple containing the cluster labels and the number of clusters determined.
    """
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def perform_clustering(
        embeddings: np.ndarray,
        dim: int,
        threshold: float,
) -> List[np.ndarray]:
    """
    Perform clustering on the embeddings by first reducing their dimensionality globally, then clustering
    using a Gaussian Mixture Model, and finally performing local clustering within each global cluster.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for UMAP reduction.
    - threshold: The probability threshold for assigning an embedding to a cluster in GMM.

    Returns:
    - A list of numpy arrays, where each array contains the cluster IDs for each embedding.
    """
    if len(embeddings) <= dim + 1:
        # Avoid clustering when there's insufficient data
        return [np.array([0]) for _ in range(len(embeddings))]

    # Global dimensionality reduction
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
    # Global clustering
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    # Iterate through each global cluster to perform local clustering
    for i in range(n_global_clusters):
        # Extract embeddings belonging to the current global cluster
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]

        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            # Handle small clusters with direct assignment
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            # Local dimensionality reduction and clustering
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        # Assign local cluster IDs, adjusting for total clusters already processed
        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    return all_local_clusters


### --- Our code below --- ###


def embed(texts):
    """
    Generate embeddings for a list of text documents.

    This function assumes the existence of an `embd` object with a method `embed_documents`
    that takes a list of texts and returns their embeddings.

    Parameters:
    - texts: List[str], a list of text documents to be embedded.

    Returns:
    - numpy.ndarray: An array of embeddings for the given text documents.
    """
    text_embeddings = own_embeddings.embed_documents(texts)
    text_embeddings_np = np.array(text_embeddings)
    return text_embeddings_np


def embed_cluster_texts(texts):
    """
    Embeds a list of texts and clusters them, returning a DataFrame with texts, their embeddings, and cluster labels.

    This function combines embedding generation and clustering into a single step. It assumes the existence
    of a previously defined `perform_clustering` function that performs clustering on the embeddings.

    Parameters:
    - texts: List[str], a list of text documents to be processed.

    Returns:
    - pandas.DataFrame: A DataFrame containing the original texts, their embeddings, and the assigned cluster labels.
    """
    text_embeddings_np = embed(texts)  # Generate embeddings
    cluster_labels = perform_clustering(
        text_embeddings_np, 10, 0.1
    )  # Perform clustering on the embeddings
    df = pd.DataFrame()  # Initialize a DataFrame to store the results
    df["text"] = texts  # Store original texts
    df["embd"] = list(text_embeddings_np)  # Store embeddings as a list in the DataFrame
    df["cluster"] = cluster_labels  # Store cluster labels
    return df


def fmt_txt(df: pd.DataFrame) -> str:
    """
    Formats the text documents in a DataFrame into a single string.

    Parameters:
    - df: DataFrame containing the 'text' column with text documents to format.

    Returns:
    - A single string where all text documents are joined by a specific delimiter.
    """
    unique_txt = df["text"].tolist()
    return "--- --- \n --- --- ".join(unique_txt)


def embed_cluster_summarize_texts(
        texts: List[str], level: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Embeds, clusters, and summarizes a list of texts. This function first generates embeddings for the texts,
    clusters them based on similarity, expands the cluster assignments for easier processing, and then summarizes
    the content within each cluster.

    Parameters:
    - texts: A list of text documents to be processed.
    - level: An integer parameter that could define the depth or detail of processing.

    Returns:
    - Tuple containing two DataFrames:
      1. The first DataFrame (`df_clusters`) includes the original texts, their embeddings, and cluster assignments.
      2. The second DataFrame (`df_summary`) contains summaries for each cluster, the specified level of detail,
         and the cluster identifiers.
    """

    # Embed and cluster the texts, resulting in a DataFrame with 'text', 'embd', and 'cluster' columns
    df_clusters = embed_cluster_texts(texts)

    # Prepare to expand the DataFrame for easier manipulation of clusters
    expanded_list = []

    # Expand DataFrame entries to document-cluster pairings for straightforward processing
    for index, row in df_clusters.iterrows():
        for cluster in row["cluster"]:
            expanded_list.append(
                {"text": row["text"], "embd": row["embd"], "cluster": cluster}
            )

    # Create a new DataFrame from the expanded list
    expanded_df = pd.DataFrame(expanded_list)

    # Retrieve unique cluster identifiers for processing
    all_clusters = expanded_df["cluster"].unique()

    print(f"--Generated {len(all_clusters)} clusters--")

    # Summarization
    template = """Here is a sub-set of LangChain Expression Langauge doc. 

    LangChain Expression Langauge provides a way to compose chain in LangChain.

    Give a detailed summary of the documentation provided.

    Documentation:
    {context}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | own_llm | StrOutputParser()

    # Format text within each cluster for summarization
    summaries = []
    for i in all_clusters:
        df_cluster = expanded_df[expanded_df["cluster"] == i]
        formatted_txt = fmt_txt(df_cluster)
        summaries.append(chain.invoke({"context": formatted_txt}))

    # Create a DataFrame to store summaries with their corresponding cluster and level
    df_summary = pd.DataFrame(
        {
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster": list(all_clusters),
        }
    )

    return df_clusters, df_summary


def recursive_embed_cluster_summarize(
        texts: List[str], level: int = 1, n_levels: int = 3
) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Recursively embeds, clusters, and summarizes texts up to a specified level or until
    the number of unique clusters becomes 1, storing the results at each level.

    Parameters:
    - texts: List[str], texts to be processed.
    - level: int, current recursion level (starts at 1).
    - n_levels: int, maximum depth of recursion.

    Returns:
    - Dict[int, Tuple[pd.DataFrame, pd.DataFrame]], a dictionary where keys are the recursion
      levels and values are tuples containing the clusters DataFrame and summaries DataFrame at that level.
    """
    results = {}  # Dictionary to store results at each level

    # Perform embedding, clustering, and summarization for the current level
    df_clusters, df_summary = embed_cluster_summarize_texts(texts, level)

    # Store the results of the current level
    results[level] = (df_clusters, df_summary)

    # Determine if further recursion is possible and meaningful
    unique_clusters = df_summary["cluster"].nunique()
    if level < n_levels and unique_clusters > 1:
        # Use summaries as the input texts for the next level of recursion
        new_texts = df_summary["summaries"].tolist()
        next_level_results = recursive_embed_cluster_summarize(
            new_texts, level + 1, n_levels
        )

        # Merge the results from the next level into the current results dictionary
        results.update(next_level_results)

    return results


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    # # ************************  build index *****************************
    # # ============= 测试效果数据集 ===================
    # # LCEL docs
    # url = "https://python.langchain.com/docs/expression_language/"
    # loader = RecursiveUrlLoader(
    #     url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text
    # )
    # docs = loader.load()
    #
    # # LCEL w/ PydanticOutputParser (outside the primary LCEL docs)
    # url = "https://python.langchain.com/docs/modules/model_io/output_parsers/quick_start"
    # loader = RecursiveUrlLoader(
    #     url=url, max_depth=1, extractor=lambda x: Soup(x, "html.parser").text
    # )
    # docs_pydantic = loader.load()
    #
    # # LCEL w/ Self Query (outside the primary LCEL docs)
    # url = "https://python.langchain.com/docs/modules/data_connection/retrievers/self_query/"
    # loader = RecursiveUrlLoader(
    #     url=url, max_depth=1, extractor=lambda x: Soup(x, "html.parser").text
    # )
    # docs_sq = loader.load()
    #
    # # Doc texts
    # docs.extend([*docs_pydantic, *docs_sq])
    # docs_texts = [d.page_content for d in docs]
    #
    # # Calculate the number of tokens for each document
    # counts = [num_tokens_from_string(d, "cl100k_base") for d in docs_texts]
    #
    # # Doc texts concat
    # d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
    # d_reversed = list(reversed(d_sorted))
    # concatenated_content = "\n\n\n --- \n\n\n".join(
    #     [doc.page_content for doc in d_reversed]
    # )
    # print(
    #     "Num tokens in all context: %s"
    #     % num_tokens_from_string(concatenated_content, "cl100k_base")
    # )
    #
    # # Doc texts split
    # chunk_size_tok = 2000
    # text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    #     chunk_size=chunk_size_tok, chunk_overlap=0
    # )
    # texts_split = text_splitter.split_text(concatenated_content)
    #
    # print(len(docs_texts))
    #
    # # Build tree
    # leaf_texts = docs_texts[:10]
    # results = recursive_embed_cluster_summarize(leaf_texts, level=1, n_levels=3)
    #
    # chroma_vectorstore = Chroma(persist_directory='D:/chroma_vectorstore')
    # # Initialize all_texts with leaf_texts
    # all_texts = leaf_texts.copy()
    #
    # # Iterate through the results to extract summaries from each level and add them to all_texts
    # for level in sorted(results.keys()):
    #     # Extract summaries from the current level's DataFrame
    #     summaries = results[level][1]["summaries"].tolist()
    #     # Extend all_texts with the summaries from the current level
    #     all_texts.extend(summaries)
    #
    # # Now, use all_texts to build the vectorstore with Chroma
    # chroma_vectorstore.from_texts(texts=all_texts, embedding=own_embeddings)
    # # ************************  build index *****************************

    # ************************  run  rag_chain  *****************************
    #  ================ 问答效果 ======================
    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # Chain
    chroma_vectorstore = Chroma(persist_directory='D:/chroma_vectorstore', embedding_function=own_embeddings)
    retriever = chroma_vectorstore.as_retriever()
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | own_llm
            | StrOutputParser()
    )

    # Question
    print(rag_chain.invoke("How to define a RAG chain? Give me a specific code example."))
