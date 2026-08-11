import os
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


def main():
    # 1. Fetch Groq API Key from environment
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is missing!")

    # Prevents underlying OpenAI SDK from raising missing key errors
    os.environ["OPENAI_API_KEY"] = groq_api_key

    # 2. Instantiate Groq LLM and assign globally
    groq_llm = get_academic_llm(api_key=groq_api_key)
    Settings.llm = groq_llm

    # 3. Configure Local Embedding Model
    data_path = os.path.join("src", "data")
    Settings.embed_model = OpenAILikeEmbedding(
        model_name="text-embedding-ada-002",
        api_base="http://localhost:1234/v1",
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
        condense_llm=groq_llm,  # <--- Ensures Groq handles history rewriting too
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
    print("     Type 'exit' or 'quit' to terminate the session.        ")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("\nEnding chat session. Goodbye!")
                break

            raw_response = chat_engine.chat(user_input)
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
    main()