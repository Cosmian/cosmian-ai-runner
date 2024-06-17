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
        llm_chain = RagLLMChain(model)
        self.db = VectorDB(
            sentence_transformer=sentence_transformer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_results=max_results
        )
        self.rag_chain = {
                             "context": self.db.as_retriever(),
                             "query": RunnablePassthrough()
                         } | llm_chain

    def add_document(self, document_path: str):
        self.db.insert(document_path)

    def invoke(self, query: Input, config: Optional[RunnableConfig] = None) -> Output:
        start_time = time.perf_counter()
        output = self.rag_chain.invoke(query, config)
        print("QUERY", query)
        print("CONFIG", config)
        print("OUTPUT", output)
        end_time = time.perf_counter()
        output['total_time'] = end_time - start_time
        return output

    def runnable(self) -> RunnableSerializable[Any, dict[str, Any]]:
        return self.rag_chain


if __name__ == "__main__":
    import logging
    import sys
    import warnings

    model = ModelValue(model_id = "llmware/dragon-yi-6b-v0", file = None, prompt = "<human>:\n{text} <|end|>\n<bot>:", task = "text2text-generation", kwargs= {
                "max_new_tokens": 256,
                "temperature": 0.01,
                "context_length": 4096,
                "repetition_penalty": 1.1,
                "gpu_layers": 0,
                'trust_remote_code': True
            })

    # Silence, please!
    logging.getLogger().setLevel(logging.ERROR)
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    if is_gpu_available():
        print("GPU is available.")

    # Choose the model
    # if len(sys.argv) > 1:
    #     model_id = sys.argv[1]
    #     model = ModelValue[model_id]
    #     if model is None:
    #         print(f"Model {model_id} not found.")
    #         sys.exit(1)
    # elif is_gpu_available():
    #     model = Model.MIXTRAL_8x7B
    # else:
    #     model = Model.DRAGON_MISTRAL_7B_V0_Q5
    print(f"Using LLM: {model.model_id}")

    # Choose the sentence transformer
    if len(sys.argv) > 2:
        sentence_transformer_id = sys.argv[2]
        sentence_transformer = SentenceTransformer[sentence_transformer_id]
        if sentence_transformer is None:
            print(f"Sentence transformer {sentence_transformer_id} not found.")
            sys.exit(1)
    else:
        sentence_transformer = SentenceTransformer.ALL_MINILM_L12_V2
    print(f"Using sentence transformer: {sentence_transformer.name}")

    rag = Rag(model=model, sentence_transformer=sentence_transformer)
    print("RAG created.")
    sources = [
        "data/Victor_Hugo_Notre-Dame_De_Paris_en.epub",
        # "data/Victor_Hugo_Les_Miserables_Fantine_1_of_5_en.epub",
        # "data/Victor_Hugo_Ruy_Blas_fr.epub",
    ]
    for source in sources:
        print(f"Loading {source}...")
        rag.add_document(source)
    print("RAG populated.")
    print("\n")
    previous_request = "Say Hello!"
    previous_context: list[Document] = []
    while True:
        request = input("Ask a question or /? for help: ")
        if request.strip() == "/?":
            print("Type a question to ask the RAG or use the following commands:")
            print("/exit: exit the program")
            print("/source: show the source of the previous response")
            print("/models: show the available models. Pass the name on the command line on next start.")
            print("/?: show this help")
            continue
        if request.strip() == "/exit":
            print("Goodbye!")
            break
        if request.strip() == "/source":
            for doc in previous_context:
                score = doc.metadata.get('score')
                if score is not None:
                    print(f"Score: {doc.metadata['score']}")
                print(f"Source: {doc.metadata['source']}")
                print(f"{doc.page_content}\n\n")
            continue
        if request.strip() == "/models":
            print("Available models:")
            for model in Model:
                print(f"  {model.name}")
            continue
        if request.strip().startswith('/'):
            print("Command not found, type /? for help")
            continue
        if request == "":
            request = previous_request
        print("...querying...")
        response = rag.invoke(request)
        previous_request = response['query']
        previous_context = response['context']
        score = response.get('score')
        if score is not None:
            print(f"Score: {response['score']}%. ", end="")
        total_time = response['total_time']
        llm_time = response['llm_time']
        print(f"Total time: {total_time:.2f} seconds (LLM: {llm_time:.2f} s)")
        print(response['text'])
        print("\n\n")
