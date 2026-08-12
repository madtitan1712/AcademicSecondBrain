import threading
import os
from typing import List, Dict, Any, Union
from pathlib import Path
from llama_index.core import VectorStoreIndex
from llama_index.retrievers.bm25 import BM25Retriever

from src.rag.ingestion.reader import load_documents_from_path
from src.rag.ingestion.ingestion import run_ingestion

ingest_lock = threading.Lock()


def ingest_new_documents(
        file_paths: List[Union[str, Path]],
        index: VectorStoreIndex,
        persist_dir: str = None
) -> Dict[str, Any]:
    """
    Dynamically ingests new documents into the live VectorStoreIndex.
    Force-inserts everything without checking for duplicates.
    """
    if persist_dir is None:
        persist_dir = os.getenv("PERSIST_DIR", "./storage")

    with ingest_lock:
        # 1. Load Documents
        documents = load_documents_from_path(file_paths)

        # 2. Run ingestion (standard pipeline, no docstore attached for dedup)
        all_nodes, leaf_nodes = run_ingestion(documents)

        # 3. Native Runtime Insert: Inject vectors and docstore nodes live
        index.docstore.add_documents(all_nodes)
        index.insert_nodes(leaf_nodes)

        # 4. Save to disk so it survives a server restart
        index.storage_context.persist(persist_dir=persist_dir)

        # 5. Rebuild BM25 Retriever with the newly expanded docstore
        bm25_retriever = BM25Retriever.from_defaults(
            docstore=index.docstore,
            similarity_top_k=12
        )

        return {
            "status": "success",
            "message": f"Successfully ingested file(s).",
            "added_total_nodes": len(all_nodes),
            "added_leaf_nodes": len(leaf_nodes),
            "bm25_retriever": bm25_retriever
        }