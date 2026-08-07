from typing import List

import chromadb
from llama_index.core import VectorStoreIndex,StorageContext
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.chroma import ChromaVectorStore


def create_hierarchical_index(all_nodes: List[BaseNode], leaf_nodes: List[BaseNode], db_path:str= './chroma_db') -> VectorStoreIndex:
    #Initialize DB and created a collection
    db=chromadb.PersistentClient(path=db_path)
    chroma_collection=db.get_or_create_collection('test_collection')
    vector_store=ChromaVectorStore(chroma_collection=chroma_collection) #Creating a ChromaVectorStore with LlamaIndex's Wrapper
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    storage_context.docstore.add_documents(all_nodes)
    index=VectorStoreIndex(nodes=leaf_nodes,storage_context=storage_context)
    return index
