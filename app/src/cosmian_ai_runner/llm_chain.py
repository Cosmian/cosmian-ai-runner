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
    def __init__(self, model_id: str, file: Any, prompt: Any, task: str, kwargs: Any):
        self.model_id = model_id
        self.file = file
        self.prompt = prompt
        self.task = task
        self.kwargs = kwargs


def __load_hf_model__(model_id: str, task: str, kwargs) -> BaseLLM:
    """
    Load a model using HuggingFace pipeline
    :param model_id:
    :return: the model as a Langchain BaseLLM
    """
    from langchain_huggingface import HuggingFacePipeline
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        pipeline,
    )

    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=hf_token, model_max_length=512
    )
    device = (
        "cuda"
        if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    if task == "text-generation":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if device == "cuda" else None,
            trust_remote_code=True,
            device_map="cuda:0" if device == "cuda" else None,
            token=hf_token,
        )
    elif task in ("text2text-generation", "summarization", "translation"):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if device == "cuda" else None,
            trust_remote_code=True,
            device_map="cuda:0" if device == "cuda" else None,
            token=hf_token,
        )
    else:
        raise ValueError(f"Got invalid task {task}, ")
    # model.to(device)
    pipe = pipeline(task, model=model, tokenizer=tokenizer, **kwargs)
    base_llm = HuggingFacePipeline(pipeline=pipe)
    return base_llm


def __load_hf_gguf_model__(model_id: str, model_file: str, config) -> BaseLLM:
    """
    Load a Huggingface model with GGUF quantization using the CTransformers LLM
    :param model_id: The Huggingface model ID
    :param model_file: the model file to use
    :return: the model as a Langchain BaseLLM
    """

    from langchain_community.llms import CTransformers

    try:
        base_llm = CTransformers(model=model_id, model_file=model_file, config=config)
        if is_gpu_available:
            from accelerate import Accelerator

            accelerator = Accelerator()
            base_llm, config = accelerator.prepare(base_llm, config)
        return base_llm
    except Exception as e:
        print("Error", e)


class RagLLMChain(LLMChain):
    def __init__(self, model: ModelValue):
        model_id = model.model_id
        model_file = model.file
        initial_prompt = model.prompt
        task = model.task
        kwargs = model.kwargs
        is_gguf = (model_file is not None) and model_file.strip().lower().endswith(
            ".gguf"
        )
        if is_gguf:
            base_llm = __load_hf_gguf_model__(
                model_id=model_id, model_file=model_file, config=kwargs
            )
        else:
            base_llm = __load_hf_model__(model_id=model_id, task=task, kwargs=kwargs)
        from langchain.prompts import PromptTemplate

        template = initial_prompt
        prompt = PromptTemplate.from_template(template)
        super().__init__(llm=base_llm, prompt=prompt)

    def invoke(
        self,
        input_args: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Override on LLMChain.invoke() to add a confidence score
        """
        if "context" in input_args:
            documents: list[Document] = input_args["context"]
            text = " ".join([doc.page_content for doc in documents])
            average_score: float | None = None
            has_scores = len(documents) > 0 and all(
                doc.metadata.get("score") is not None for doc in documents
            )
            if has_scores:
                average_score = sum(doc.metadata["score"] for doc in documents) / len(
                    documents
                )
                if 0 <= average_score <= 1:
                    average_score = int(average_score * 100)
                elif average_score < -1:
                    average_score = int((average_score + 10) * 10)
        elif "text" in input_args:
            text = input_args["text"]
        else:
            raise Exception("Missing text argument.")
        # start_time = time.perf_counter()
        output = super().invoke({"text": text}, config, **kwargs)
        # output = super().invoke(input_args, config, **kwargs)
        # end_time = time.perf_counter()
        # output['llm_time'] = end_time - start_time
        # output['score'] = average_score
        return output
