import os
import time
from enum import Enum
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type

from langchain.text_splitter import CharacterTextSplitter, TextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VST, VectorStore, VectorStoreRetriever
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from .epub_loader import load_epub

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class STValue:
    def __init__(self, file: str, score_threshold: Any = None):
        self.file = file
        self.score_threshold = score_threshold


class FilteredRetriever(VectorStoreRetriever):
    """
    A retriever that filters the results based on a score threshold.
    The retriever will insert the score of the document in the metadata.
    """

    vectorstore: VectorStore
    score_threshold: Any = None  # the minimum similarity score to return.
    max_results: int = 4  # equivalent to the k factor in the similarity_search method

    def get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        documents: list[Document] = []
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=self.max_results
        )
        for doc, score in results:
            if self.score_threshold is None or score > self.score_threshold:
                doc.metadata["score"] = score
                documents.append(doc)
        return documents


class VectorDB(VectorStore):
    _embeddings: Embeddings
    _splitter: TextSplitter
    _db: VectorStore
    _score_threshold: Any
    _max_results: int

    def __init__(
        self,
        sentence_transformer: STValue,
        chunk_size: int = 256,
        chunk_overlap: int = 0,
        max_results=4,
    ):
        self._embeddings = HuggingFaceEmbeddings(
            model_name=sentence_transformer.file
        )
        self._score_threshold = sentence_transformer.score_threshold
        self._max_results = max_results
        self._splitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self._db = FAISS.from_texts(["dummy"], self._embeddings)

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Access the query embedding object if available."""
        return self._embeddings

    @embeddings.setter
    def embeddings(self, value: Embeddings) -> None:
        """Set the query embedding object."""
        self._embeddings = value

    @property
    def threshold(self) -> float:
        return self._score_threshold

    @property
    def max_results(self) -> int:
        return self._max_results

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Override the add_texts method to add texts to the database.
        """
        return self._db.add_texts(texts, metadatas, **kwargs)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """
        Override the similarity_search method to perform a similarity search.
        """
        return self._db.similarity_search(query, k, **kwargs)

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VST:
        """
        Create a VectorStore from a list of texts.
        Override the from_texts method to create a VectorStore from a list of texts.
        """
        return db.from_texts(texts, embedding, metadatas, **kwargs)

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        Override the _select_relevance_score_fn method to return the relevance score.
        """
        return self._db._select_relevance_score_fn()

    def similarity_search_with_score(
        self, *args: Any, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """
        Override the similarity_search_with_score method to return the results with the relevance scores.
        Perform a similarity search and return the results with the relevance scores.
        """
        return self._db.similarity_search_with_score(*args, **kwargs)

    def as_retriever(self, **kwargs: Any) -> VectorStoreRetriever:
        """
        Override the as_retriever method to return a FilteredRetriever.
        Return a retriever that filters the results based on a score threshold.
        :param kwargs: ignored
        :return: the retriever
        """
        return FilteredRetriever(
            vectorstore=self,
            score_threshold=self._score_threshold,
            max_results=self._max_results,
        )

    def insert(self, document_path: str):
        document = __load_document__(document_path)
        chunks = self._splitter.split_documents([document])
        self._db.add_documents(chunks)


def __load_document__(path: str) -> Document:
    if path.endswith(".epub"):
        return load_epub(path)
    raise ValueError(f"Unsupported file type: {path}")
