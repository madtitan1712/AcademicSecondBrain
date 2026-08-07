from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
def build_query_engine(index:VectorStoreIndex) ->RetrieverQueryEngine:
    #Vector Retriever from ChromaDb
    vector_retriever=index.as_retriever(similarity_top_k=25)
    #BM25_Retriever from tf-idf
    bm25_retriever=BM25Retriever.from_defaults(
        docstore=index.storage_context.docstore,
        similarity_top_k=25
    )
    #Query Fusion Retriever is hybrid retriever rather than a normal base retriever. Does both types of retrieval and stores chunks
    hybrid_retriever=QueryFusionRetriever(
        [vector_retriever,bm25_retriever],
        similarity_top_k=25,
        num_queries=1,
        mode="reciprocal_rerank",
    )
    #Merges the retrievers chunks based on the parent nodes and the child nodes
    auto_merging_retriever=AutoMergingRetriever(
        hybrid_retriever,
        storage_context=index.storage_context,
        simple_ratio_thresh=0.3
    )
    #Reranks and Query engine uses the reranker to answers to the Query
    reranker=SentenceTransformerRerank(
        model="BAAI/bge-reranker-v2-m3",
        top_n=5
    )
    query_engine=RetrieverQueryEngine.from_args(
        retriever=auto_merging_retriever,
        node_postprocessors=[reranker]
    )
    return query_engine

