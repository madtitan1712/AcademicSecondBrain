import os
import shutil
from fastapi import APIRouter, Request, HTTPException, UploadFile, File
from src.rag.registry.documents import list_documents
from src.rag.ingestion.ingester import ingest_new_documents, delete_document

router = APIRouter(prefix="/api", tags=["Documents"])

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")


@router.get("/docs")
def get_documents_endpoint(request: Request):
    """Retrieves a list of all active documents in the RAG system."""
    try:
        index = getattr(request.app.state, "index", None)
        if not index:
            raise HTTPException(status_code=503, detail="Vector index is not initialized.")

        # Query local docstore using existing helper
        raw_docs = list_documents(index)

        mapped_docs = []
        for doc in raw_docs:
            filename = doc.get("file_name", "unknown")
            mapped_docs.append({
                "document_id": filename,
                "filename": filename,
                "file_path": doc.get("file_path", "unknown"),
                "chunk_count": len(doc.get("part_ids", [])),
                "ingested_at": "unknown"
            })

        return mapped_docs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/docs/upload")
async def upload_document_endpoint(
    request: Request,
    file: UploadFile = File(...)
):
    """
    Uploads and ingests a new document (PDF, DOCX, PPTX, etc.).
    """
    try:
        index = getattr(request.app.state, "index", None)
        retriever = getattr(request.app.state, "retriever", None)

        if not index or not retriever:
            raise HTTPException(status_code=503, detail="RAG system is not initialized.")

        # 1. Ensure upload destination exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        # 2. Save incoming file stream to local storage
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 3. Ingest document into index, vector store, and docstore
        result = ingest_new_documents([file_path], index)

        # 4. Hot-swap the active BM25 retriever
        if result.get("bm25_retriever"):
            retriever.update_bm25(result["bm25_retriever"])

        return {
            "message": "Document ingested successfully",
            "document_id": file.filename,
            "metadata": {
                "added_total_nodes": result.get("added_total_nodes", 0),
                "added_leaf_nodes": result.get("added_leaf_nodes", 0),
                "file_path": file_path
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@router.delete("/docs/{document_id}")
def delete_document_endpoint(document_id: str, request: Request):
    """
    Deletes a specific document from the vector store and docstore.
    """
    try:
        index = getattr(request.app.state, "index", None)
        retriever = getattr(request.app.state, "retriever", None)

        if not index or not retriever:
            raise HTTPException(status_code=503, detail="RAG system is not initialized.")

        # Trigger the deletion process
        # Since our GET /docs maps the filename as the document_id, we pass it as file_name
        result = delete_document(
            file_name=document_id,
            index=index,
            retriever_wrapper=retriever
        )

        # If the document wasn't found in the docstore
        if result.get("status") == "error":
            raise HTTPException(status_code=404, detail=result.get("message"))

        return {
            "message": f"Document {document_id} deleted"
        }

    except HTTPException:
        # Re-raise HTTP exceptions to maintain the correct status code (e.g., 404 vs 500)
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))