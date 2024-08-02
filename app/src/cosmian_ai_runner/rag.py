"""
This module provides the Rag class for retrieval-augmented generation (RAG) using
vector stores and Hugging Face models with LangChain.
"""
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.utils import Output

from .llm_chain import LLM, ModelValue
from .vector_db import STValue, VectorDB


class Rag:
    """
    A class to perform retrieval-augmented generation (RAG) using a vector database
    and language models.
    Attributes:
        db (VectorDB): The vector database for storing and retrieving documents.
        rag_chain (RunnableSerializable[Any, dict[str, Any]]): The RAG chain for generating responses.
    """
    db: VectorDB
    rag_chain: RunnableSerializable[Any, dict[str, Any]]

    def __init__(
        self,
        sentence_transformer=STValue,
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        max_results=5,
    ):
        """
        Initialize the Rag class.
        Args:
            sentence_transformer (STValue): The sentence transformer configuration.
            chunk_size (int): The chunk size for text splitting.
            chunk_overlap (int): The chunk overlap for text splitting.
            max_results (int): The maximum number of results to return.
        """
        self.db = VectorDB(
            sentence_transformer=sentence_transformer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_results=max_results,
        )

    def add_document(self, document_path: str):
        """
        Add a document to the vector database.
        Args:
            document_path (str): The path to the document to add.
        """
        self.db.insert(document_path)

    def invoke(
        self, model: ModelValue, query: str, config: Optional[RunnableConfig] = None
    ) -> Output:
        """
        Perform a RAG invocation using the specified model and query.
        Args:
            model (ModelValue): The model configuration.
            query (str): The query string to search for.
            config (Optional[RunnableConfig]): Optional configuration for the runnable.

        Returns:
            Output: The output of the RAG invocation.
        """
        llm_chain = LLM(model=model)
        rag_chain = {
            "context": self.db.as_retriever(),
        } | llm_chain
        return rag_chain.invoke(query, config)
