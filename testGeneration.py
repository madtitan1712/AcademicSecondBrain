import os
import asyncio
from llama_index.core import Settings
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer

from src.rag.synthesis.engine import get_academic_llm
from src.rag.ingestion.reader import load_documents_from_path
from src.rag.ingestion.ingestion import run_ingestion
from src.rag.ingestion.indexer import create_hierarchical_index
from src.rag.retrieval.retreiver import build_retriever_stack
from src.rag.synthesis.engine import format_response_with_sources
from src.rag.ingestion.ingester import ingest_new_documents


async def main():
    # 1. Fetch Groq API Key from environment
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is missing!")

    os.environ["OPENAI_API_KEY"] = groq_api_key

    # 2. Instantiate Groq LLM and assign globally
    groq_llm = get_academic_llm(api_key=groq_api_key)
    Settings.llm = groq_llm

    # 3. Configure Local Embedding Model dynamically via Env
    data_path = os.path.join("src", "data")
    embed_api_base = os.getenv("EMBEDDING_API_BASE", "http://localhost:1234/v1")

    Settings.embed_model = OpenAILikeEmbedding(
        model_name="text-embedding-ada-002",
        api_base=embed_api_base,
        api_key="lm-studio",
    )

    # Ingestion & Indexing
    print("1. Loading documents & running ingestion pipeline...")
    documents = load_documents_from_path(data_path)
    all_nodes, leaf_nodes = run_ingestion(documents)

    print("\n2. Creating/Loading hierarchical index...")
    index = create_hierarchical_index(all_nodes, leaf_nodes)

    print("\n3. Building retriever and postprocessors...")
    retriever, postprocessors = build_retriever_stack(index)

    # 4. Assemble Academic Chat Engine with Groq
    print("\n4. Building academic chat engine with Groq...")
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        node_postprocessors=postprocessors,
        llm=groq_llm,
        condense_llm=groq_llm,
        memory=memory,
        system_prompt=(
            "You are a strict and precise academic research assistant. "
            "Use the provided document context and conversation history to answer user questions. "
            "Always cite relevant sources and papers when explaining methodology or findings."
        )
    )

    # 5. Interactive Chat Loop
    print("\n" + "=" * 60)
    print("        ACADEMIC SECOND BRAIN - CHAT INTERFACE READY        ")
    print("     Commands: 'exit', 'quit', or '/upload <path/to/pdf>'   ")
    print("=" * 60 + "\n")

    while True:
        try:
            # Note: For CLI scripts `input` is fine. If deploying to FastAPI, use API requests instead.
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("\nEnding chat session. Goodbye!")
                break

            # --- HOT UPLOAD COMMAND INTERCEPT ---
            if user_input.lower().startswith("/upload "):
                filepath = user_input.split(" ", 1)[1].strip()
                if not os.path.exists(filepath):
                    print(f"\n[Error] File '{filepath}' not found.")
                    continue

                print(f"\n[System] Initiating dynamic upload for {filepath}...")
                result = ingest_new_documents([filepath], index)

                print(f"Status: {result['status']}")
                print(f"Message: {result['message']}")
                print(f"Nodes Added: {result['added_leaf_nodes']} leaf / {result['added_total_nodes']} total")

                # Clean Hot-swap of the BM25 Retriever
                if result.get("bm25_retriever"):
                    retriever.update_bm25(result["bm25_retriever"])
                    print("[System] Active BM25 Retriever stack updated successfully.")

                continue
            # ------------------------------------

            # Async execution for the chat generation to prevent blocking the event loop
            raw_response = await chat_engine.achat(user_input)
            result = format_response_with_sources(raw_response)

            print("\n" + "=" * 60)
            print("                      SYNTHESIZED ANSWER                     ")
            print("=" * 60)
            print(result["answer"])

            if result.get("sources"):
                print("\n" + "-" * 60)
                print("                       RETRIEVED SOURCES                     ")
                print("-" * 60)
                for idx, src in enumerate(result["sources"], 1):
                    page_val = src.get("page") or src.get("page_number", "N/A")
                    print(f"[{idx}] File: {src['file_name']} (Page: {page_val}) | Score: {src['score']}")
                    print(f"    Snippet: {src['snippet']}\n")

        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError processing chat query: {e}")


if __name__ == "__main__":
    asyncio.run(main())