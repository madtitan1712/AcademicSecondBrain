from typing import Dict, Any, Optional
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from src.rag.registry.sessions import get_or_create_session, save_session
from src.rag.synthesis.engine import format_response_with_sources


async def handle_stateless_chat(
        message: str,
        retriever: Any,
        llm: Any,
        session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Stateless chat invocation:
    1. Loads or creates session memory.
    2. Dynamically builds the ChatEngine.
    3. Streams/waits for response.
    4. Saves state and returns.
    """

    # 1. Load Session
    active_session_id, memory = get_or_create_session(session_id)

    # 2. Build Engine Fresh
    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        llm=llm,
        condense_llm=llm,
        memory=memory,
        system_prompt=(
            "You are a strict and precise academic research assistant. "
            "Use the provided document context and conversation history to answer user questions. "
            "Always cite relevant sources and papers when explaining methodology or findings."
        )
    )

    # 3. Generate Response
    raw_response = await chat_engine.achat(message)
    formatted_result = format_response_with_sources(raw_response)

    # 4. Save state back to SQLite
    save_session(active_session_id, memory)

    return {
        "session_id": active_session_id,
        "answer": formatted_result["answer"],
        "sources": formatted_result["sources"]
    }