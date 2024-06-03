import time
from typing import Any, Optional

from langchain_core.runnables import (
    RunnableConfig,
    RunnablePassthrough,
    RunnableSerializable,
)
from langchain_core.runnables.utils import Input, Output

from .llm_chain import ModelValue, RagLLMChain
from .vector_db import SentenceTransformer, VectorDB


class Rag:
    # db: VectorDB
    rag_chain: RunnableSerializable[Any, dict[str, Any]]

    def __init__(
        self,
        model: ModelValue,
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        max_results=5,
    ):
        llm_chain = RagLLMChain(model)
        # self.db = VectorDB(
        #     sentence_transformer=sentence_transformer,
        #     chunk_size=chunk_size,
        #     chunk_overlap=chunk_overlap,
        #     max_results=max_results
        # )
        self.rag_chain = {"context": {}, "query": RunnablePassthrough()} | llm_chain

    # def add_document(self, document_path: str):
    #     self.db.insert(document_path)

    def invoke(self, query: Input, config: Optional[RunnableConfig] = None) -> Output:
        start_time = time.perf_counter()
        output = self.rag_chain.invoke(query, config)
        end_time = time.perf_counter()
        output["total_time"] = end_time - start_time
        return output

    def runnable(self) -> RunnableSerializable[Any, dict[str, Any]]:
        return self.rag_chain
