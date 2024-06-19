# Follows https://python.langchain.com/docs/integrations/llms/huggingface_pipelines for the Huggingface pipeline with
# LangChain
import time
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (RunnableConfig, RunnablePassthrough,
                                      RunnableSerializable)
from langchain_core.runnables.utils import Input, Output

from .detect import is_gpu_available
from .llm_chain import ModelValue, RagLLMChain
from .vector_db import SentenceTransformer, VectorDB


class Rag:
    db: VectorDB
    rag_chain: RunnableSerializable[Any, dict[str, Any]]

    def __init__(self,
                 model: ModelValue,
                 sentence_transformer=SentenceTransformer.ALL_MINILM_L12_V2,
                 chunk_size: int = 512,
                 chunk_overlap: int = 0,
                 max_results=5
                 ):
        llm_chain = RagLLMChain(model=model)
        self.db = VectorDB(
            sentence_transformer=sentence_transformer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_results=max_results
        )
        self.rag_chain = {
                             "context": self.db.as_retriever(),
                         } | llm_chain

    def add_document(self, document_path: str):
        self.db.insert(document_path)

    def invoke(self, query: str, config: Optional[RunnableConfig] = None) -> Output:
        return self.rag_chain.invoke(query, config)

    def runnable(self) -> RunnableSerializable[Any, dict[str, Any]]:
        return self.rag_chain
