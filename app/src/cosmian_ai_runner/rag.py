# Follows https://python.langchain.com/docs/integrations/llms/huggingface_pipelines for the Huggingface pipeline with
# LangChain
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.utils import Output

from .llm_chain import ModelValue, RagLLMChain
from .vector_db import STValue, VectorDB


class Rag:
    db: VectorDB
    rag_chain: RunnableSerializable[Any, dict[str, Any]]

    def __init__(
        self,
        sentence_transformer=STValue,
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        max_results=5,
    ):
        self.db = VectorDB(
            sentence_transformer=sentence_transformer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_results=max_results,
        )

    def add_document(self, document_path: str):
        self.db.insert(document_path)

    def invoke(
        self, model: ModelValue, query: str, config: Optional[RunnableConfig] = None
    ) -> Output:
        llm_chain = RagLLMChain(model=model)
        rag_chain = {
            "context": self.db.as_retriever(),
        } | llm_chain
        return rag_chain.invoke(query, config)

    def runnable(self) -> RunnableSerializable[Any, dict[str, Any]]:
        return self.rag_chain
