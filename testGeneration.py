import os

from llama_index.llms.openai_like import OpenAILike

from src.rag.ingestion.reader import load_documents_from_path
from src.rag.ingestion.ingestion import run_ingestion
from src.rag.ingestion.indexer import create_hierarchical_index
from src.rag.retrieval.retreiver import build_query_engine
from llama_index.core import Settings
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from src.rag.synthesis.engine import (
    build_academic_query_engine,
    format_response_with_sources
)

def main():
    # Define data directory path
    data_path = os.path.join("src", "data")
    Settings.embed_model = OpenAILikeEmbedding(
        model_name="text-embedding-ada-002",
        api_base="http://localhost:1234/v1",
        api_key="lm-studio",
    )

    # Step 1: Load documents & run ingestion pipeline
    print("1. Loading documents & running ingestion pipeline...")
    documents = load_documents_from_path(data_path)
    all_nodes, leaf_nodes = run_ingestion(documents)
    print(f"   Loaded {len(documents)} document(s) -> Generated {len(all_nodes)} total nodes ({len(leaf_nodes)} leaf nodes).")

    # Step 2: Create or load vector index
    print("\n2. Creating/Loading hierarchical index...")
    index = create_hierarchical_index(all_nodes, leaf_nodes)

    # Step 3: Build AutoMerging Hybrid Retriever and Reranker
    print("\n3. Building retriever and postprocessors...")
    retriever, postprocessors = build_query_engine(index)

    # Step 4: Assemble Academic Query Engine
    print("\n4. Building academic query engine...")
    query_engine = build_academic_query_engine(
        retriever=retriever,
        node_postprocessors=postprocessors
    )

    # Step 5: Query the engine and print formatted results
    test_query = "What are the core methodology and main findings discussed in these documents?"
    print(f"\n5. Executing Query: '{test_query}'\n")

    raw_response = query_engine.query(test_query)
    result = format_response_with_sources(raw_response)

    # Display Answer
    print("=" * 60)
    print("                      SYNTHESIZED ANSWER                     ")
    print("=" * 60)
    print(result["answer"])

    # Display Sources
    print("\n" + "=" * 60)
    print("                       RETRIEVED SOURCES                     ")
    print("=" * 60)
    for idx, src in enumerate(result["sources"], 1):
        print(f"[{idx}] File: {src['file_name']} (Page: {src['page_number']}) | Score: {src['score']}")
        print(f"    Snippet: {src['snippet']}\n")

if __name__ == "__main__":
    main()