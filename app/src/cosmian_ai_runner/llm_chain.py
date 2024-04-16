import os
import time
from enum import Enum
from typing import Any, Dict, Optional

import torch
from langchain.chains import LLMChain
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from langchain_core.runnables import RunnableConfig

from .detect import is_gpu_available

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


class ModelValue:
    def __init__(self, model_id: str, file: Any, prompt: Any):
        self.model_id = model_id
        self.file = file
        self.prompt = ""

# DRAGON_MISTRAL_ANSWER_TOOL = ModelValue(model_id="llmware/dragon-mistral-answer-tool", file="dragon-mistral.gguf", prompt=None)
# DRAGON_YI_6B_V0 = ModelValue(model_id="llmware/dragon-yi-6b-v0", file="dragon-yi-6b-q4_k_m.gguf", prompt=None)
# DRAGON_YI_6B_V0_GGUF = ModelValue(
#     model_id="TheBloke/dragon-yi-6B-v0-GGUF",
#     file="dragon-yi-6b-v0.Q4_K_M.gguf",
#     prompt=None
# )
# DRAGON_MISTRAL_7B_V0_Q4 = ModelValue(
#     model_id="llmware/dragon-mistral-7b-v0",
#     file="dragon-mistral-7b-q4_k_m.gguf",
#     prompt=None
# )
# DRAGON_MISTRAL_7B_V0_Q5 = ModelValue(
#     model_id="TheBloke/dragon-mistral-7B-v0-GGUF",
#     file="dragon-mistral-7b-v0.Q5_K_M.gguf",
#     prompt=None
# )
# MIXTRAL_8x7B = ModelValue(model_id="mistralai/Mixtral-8x7B-Instruct-v0.1", file=None,
#                               prompt="""Use the following pieces of context to answer the user question.
# This context retrieved from a knowledge base and you should use only the facts from the context to answer.
# Your answer must be based on the context. If the context not contain the answer, just say that 'I don't know',
# don't try to make up an answer, use the context.
# Don't address the context directly, but use it to answer the user question like it's your own knowledge.
# Answer in short, use up to 10 words.

# Context:
# {context}

# Question: {query}.
# """)


def __load_hf_model__(model_id: str) -> BaseLLM:
    """
    Load a model using HuggingFace pipeline
    :param model_id:
    :return: the model as a Langchain BaseLLM
    """
    from langchain_community.llms.huggingface_pipeline import \
        HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if is_gpu_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            device_map="cuda:0",
        )
        model.to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        device_map="auto",
        pad_token_id=tokenizer.eos_token_id
    )
    base_llm = HuggingFacePipeline(pipeline=pipe)
    return base_llm


def __load_hf_gguf_model__(model_id: str, model_file: str = None) -> BaseLLM:
    """
    Load a Huggingface model with GGUF quantization using the CTransformers LLM
    :param model_id: The Huggingface model ID
    :param model_file: the model file to use
    :return: the model as a Langchain BaseLLM
    """

    # see: https://python.langchain.com/docs/integrations/providers/ctransformers
    from langchain_community.llms import CTransformers

    # parameters available: https://github.com/marella/ctransformers#config
    config = {
        'max_new_tokens': 256,
        'temperature': 0.01,
        'context_length': 4096,
        'repetition_penalty': 1.1,
        'gpu_layers': 50 if is_gpu_available() else 0,
    }
    base_llm = CTransformers(model=model_id, model_file=model_file, config=config)
    if is_gpu_available():
        from accelerate import Accelerator
        accelerator = Accelerator()
        base_llm, config = accelerator.prepare(base_llm, config)
    return base_llm


class RagLLMChain(LLMChain):

    def __init__(self, model: ModelValue, **kwargs: Any):
        model_id = model.model_id
        model_file = model.file
        prompt = model.prompt
        is_gguf = (model_file is not None) and model_file.strip().lower().endswith('.gguf')
        if is_gguf:
            base_llm = __load_hf_gguf_model__(model_id=model_id, model_file=model_file)
        else:
            base_llm = __load_hf_model__(model_id=model_id)
        from langchain.prompts import PromptTemplate
        if prompt is not None:
            print("Using a custom prompt")
            template = prompt
        else:
            template = """<human>: Use the following pieces of context to answer the user question.
This context retrieved from a knowledge base and you should use only the facts from the context to answer.
Your answer must be based on the context. If the context not contain the answer, just say that 'I don't know',
don't try to make up an answer, use the context.
Don't address the context directly, but use it to answer the user question like it's your own knowledge.
Answer in short, use up to 10 words.

Context:
{context}

Question: {query}
<bot>:
"""
        prompt = PromptTemplate.from_template(template)
        super().__init__(llm=base_llm, prompt=prompt, **kwargs)

    def invoke(
            self,
            input_args: Dict[str, Any],
            config: Optional[RunnableConfig] = None,
            **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Override on LLMChain.invoke() to add a confidence score
        """
        documents: list[Document] = input_args['context']
        average_score: float | None = None
        has_scores = len(documents) > 0 and all(doc.metadata.get('score') is not None for doc in documents)
        if has_scores:
            average_score = sum(doc.metadata['score'] for doc in documents) / len(documents)
            if 0 <= average_score <= 1:
                average_score = int(average_score * 100)
            elif average_score < -1:
                average_score = int((average_score + 10) * 10)
        start_time = time.perf_counter()
        output = super().invoke(input_args, config, **kwargs)
        end_time = time.perf_counter()
        output['llm_time'] = end_time - start_time
        output['score'] = average_score
        return output
