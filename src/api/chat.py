from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

from src.api.schemas import ChatRequest
from src.rag.synthesis.chat import handle_streaming_chat

router = APIRouter(tags=["Chat"])


@router.post("/chat")
async def chat_stream_endpoint(request_data: ChatRequest, request: Request):
    """
    Streamable chat endpoint returning Server-Sent Events (SSE).
    """
    try:
        # Access shared state from app context
        retriever = request.app.state.retriever
        llm = request.app.state.llm
        postprocessors = request.app.state.postprocessors

        if not retriever or not llm:
            raise HTTPException(status_code=503, detail="RAG pipeline is not initialized yet.")

        event_generator = handle_streaming_chat(
            message=request_data.question,
            retriever=retriever,
            llm=llm,
            session_id=request_data.session_id,
            node_postprocessors=postprocessors
        )

        return StreamingResponse(
            event_generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable buffering in reverse proxies like Nginx
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))