# -*- coding: utf-8 -*-
import os
from http import HTTPStatus
from typing import Dict

import torch
from asgiref.wsgi import WsgiToAsgi
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, pipeline)

from .auth import check_token
from .config import AppConfig
from .detect import is_gpu_available
from .llm_chain import ModelValue, RagLLMChain
from .rag import Rag
from .summarizer import Summarizer
from .translator import Translator
from .vector_db import SentenceTransformer, VectorDB

torch.set_num_threads(os.cpu_count() or 1)

# DRAGON_MISTRAL_ANSWER_TOOL = ModelValue("llmware/dragon-mistral-answer-tool", None)
# DRAGON_YI_6B_V0 = ModelValue("llmware/dragon-yi-6b-v0", None)
DRAGON_YI_6B_V0_GGUF = ModelValue(
    model_id = "TheBloke/dragon-yi-6B-v0-GGUF",
    file = "dragon-yi-6b-v0.Q4_K_M.gguf",
    task = "text2text-generation",
    prompt = "",
    kwargs = {
        "max_new_tokens": 256,
        "temperature": 0.01,
        "context_length": 4096,
        "repetition_penalty": 1.1,
        "gpu_layers": 0
    }
)

app = Flask(__name__)
app_asgi = WsgiToAsgi(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

config_path = os.getenv("CONFIG_PATH", "config.json")
with open(config_path) as f:
    app_config = AppConfig.load(f)

summarizer_by_lang: Dict[str, Summarizer] = {
    lang: Summarizer(**model_config)
    for lang, model_config in app_config["summary"].items()
}
translator = Translator(**app_config["translation"])

model_values = {}
data_list = AppConfig.get_models_config()
for item in data_list:
    model_id = item.get("model_id")
    file = item.get("file")
    prompt = item.get("prompt")
    task = item.get("task")
    kwargs = item.get("kwargs")
    model_value = ModelValue(model_id, file, prompt, task, kwargs)
    model_values[model_id] = model_value

# Create RAG
# rag_model = DRAGON_YI_6B_V0_GGUF
sentence_transformer = SentenceTransformer.ALL_MINILM_L12_V2

# rag = Rag(model=rag_model, sentence_transformer=sentence_transformer)
# print("RAG created.")
# sources = [
#     "data/Victor_Hugo_Notre-Dame_De_Paris_en.epub",
#     # "data/Victor_Hugo_Les_Miserables_Fantine_1_of_5_en.epub",
#     # "data/Victor_Hugo_Ruy_Blas_fr.epub",
# ]
# for source in sources:
#     print(f"Loading {source}...")
#     rag.add_document(source)
# print("RAG populated.")
# print("\n")


@app.post("/summarize")
@check_token()
async def post_summarize():
    if "doc" not in request.form:
        return ("Error: Missing file content", 400)

    text = request.form["doc"]
    src_lang = request.form.get("src_lang", default="default")

    try:
        summarizer = summarizer_by_lang.get(src_lang, summarizer_by_lang["default"])
        summary = summarizer(text)
    except ValueError as e:
        return (str(e), 400)

    return jsonify(
        {
            "summary": summary,
        }
    )


@app.post("/translate")
@check_token()
async def post_translate():
    if "doc" not in request.form:
        return ("Error: Missing file content", 400)
    if "src_lang" not in request.form:
        return ("Error: Missing source language", 400)
    if "tgt_lang" not in request.form:
        return ("Error: Missing target language", 400)

    text = request.form["doc"]
    src_lang = request.form["src_lang"]
    tgt_lang = request.form["tgt_lang"]

    try:
        result = translator(text, src_lang=src_lang, tgt_lang=tgt_lang)
    except ValueError as e:
        return (f"Error: {e}", 400)

    return jsonify(
        {
            "translation": result,
        }
    )


@app.get("/health")
def health_check():
    """Health check of the application."""
    return Response(response="OK", status=HTTPStatus.OK)


@app.post("/predict")
@check_token()
async def make_predictionl():
    """Make a prediction using selected model."""
    if "text" not in request.form:
        return ("Error: Missing text content", 400)
    if "model" not in request.form:
        return ("Error: Missing selected model", 400)

    text = request.form["text"]
    model_name = request.form["model"]

    if is_gpu_available():
        print("GPU is available.")

    # Choose the model
    if model_name:
        try:
            model = model_values[model_name]
        except KeyError as e:
            return (f"Error model not found: {e}", 404)

    print(f"Using LLM: {model.model_id}")
    try:
        llm = RagLLMChain(model=model)
        print("LLM created.")
        response = llm.invoke({"text": text})
        return jsonify(
            {
                "response": response,
            }
        )
    except Exception as e:
        return (f"Error during prediction: {e}", 400)


@app.get("/models")
@check_token()
async def list_models():
    """List all the configured models."""
    return jsonify(
        {
            "models": data_list,
        }
    )

@app.post("/rag")
@check_token()
async def build_rag():
    if "text" not in request.form:
        return ("Error: Missing text content", 400)

    question = request.form["text"]
    print("TEXT", question)

    if is_gpu_available():
        print("GPU is available.")

    # response = rag.invoke(text)
    db = VectorDB(
        sentence_transformer=sentence_transformer,
        chunk_size=512,
        chunk_overlap=0,
        max_results=5
    )
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    model_name = "HuggingFaceH4/zephyr-7b-beta"
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    # )
    print("MODEL")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    print("TOKENIZER")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # llm = HuggingFacePipeline.from_model_id(
    #     model_id=model_name,
    #     task="text-generation",
    #     pipeline_kwargs={
    #         "temperature":0.2,
    #         "do_sample":True,
    #         "repetition_penalty":1.1,
    #         "return_full_text":True,
    #         "max_new_tokens":400,
    #     },
    # )
    print("TEXT GEN")
    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=400,
    )
    print("LLM")

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    prompt_template = """
    <|system|>
    Answer the question based on your knowledge. Use the following context to help:

    {context}

    </s>
    <|user|>
    {question}
    </s>
    <|assistant|>

    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    print("CHAIN")
    llm_chain = prompt | llm | StrOutputParser()
    retriever = db.as_retriever()

    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain

    print("INVOKE")
    print(retriever)
    response = llm_chain.invoke({"context": "", "question": question})
    # response = rag_chain.invoke(question)


    print("RESPONSE", response)
    return jsonify(
        {
            "response": response,
        }
    )
