from fastapi import APIRouter, HTTPException
from datetime import datetime
from src.rag.registry.sessions import list_sessions, get_session_history, delete_session

router = APIRouter(tags=["Sessions"])


@router.get("/api/chat/sessions")
def get_sessions_endpoint():
    """Gets all historical chat sessions for the sidebar keys."""
    try:
        sessions = list_sessions()

        mapped_sessions = []
        for s in sessions:
            session_id = s["session_id"]

            # 1. Fetch the history for this specific session
            history = get_session_history(session_id)

            # 2. Set a default fallback title
            title = f"New Chat ({session_id[:4]})"

            # 3. Scan for the first user message to use as the title
            for msg in history:
                if msg["role"] == "user":
                    content = msg["content"].strip()
                    words = content.split()

                    # Truncate to the first 5-6 words for a clean sidebar look
                    if len(words) > 6:
                        title = " ".join(words[:6]) + "..."
                    else:
                        title = content
                    break  # Stop looking once we find the first user message

            mapped_sessions.append({
                "id": session_id,
                "title": title,
                "created_at": s["created_at"]
            })

        return mapped_sessions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/chat/history/{session_id}")
def get_session_history_endpoint(session_id: str):
    """Retrieves the full message history for a specific session."""
    try:
        history = get_session_history(session_id)

        mapped_history = []
        for idx, msg in enumerate(history):
            mapped_history.append({
                "id": idx + 1,
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": datetime.utcnow().isoformat()
            })
        return mapped_history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/chat/session/{session_id}")
def delete_session_endpoint(session_id: str):
    """Deletes a chat session and its history."""
    try:
        delete_session(session_id)
        return {"message": "Session deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))