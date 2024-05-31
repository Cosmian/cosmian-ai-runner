# -*- coding: utf-8 -*-
import os
from http import HTTPStatus
from typing import Dict

import torch
from asgiref.wsgi import WsgiToAsgi
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from .auth import check_token
from .config import AppConfig
from .detect import is_gpu_available
from .llm_chain import ModelValue, RagLLMChain
from .rag import Rag
from .summarizer import Summarizer
from .translator import Translator
from .vector_db import SentenceTransformer, VectorDB

torch.set_num_threads(os.cpu_count() or 1)

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
    model_value = ModelValue(model_id, file, prompt, task)
    model_values[model_id] = model_value

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
    # if "sentence_transformer" not in request.form:
    #     return ("Error: Missing sentence transformer", 400)

    text = request.form["text"]
    model_name = request.form["model"]
    # sentence_transformer_name = request.form["sentence_transformer"]

    if is_gpu_available():
        print("GPU is available.")

    # Choose the model
    if model_name:
        model = model_values[model_name]
        if model is None:
            print(f"Model {model_name} not found.")
            return jsonify(
                {
                    "request": "KO",
                }
            )
    # elif is_gpu_available():
    #     model = Model.MIXTRAL_8x7B
    # else:
    #     model = Model.DRAGON_MISTRAL_7B_V0_Q5
    print(f"Using LLM: {model.model_id}")

    # Choose the sentence transformer
    # if sentence_transformer_name:
    #     sentence_transformer = SentenceTransformer[sentence_transformer_name]
    #     if sentence_transformer is None:
    #         print(f"Sentence transformer {sentence_transformer_name} not found.")
    #         return jsonify(
    #             {
    #                 "request": "KO",
    #             }
    #         )
    # else:
    #     sentence_transformer = SentenceTransformer.ALL_MINILM_L12_V2
    # print(f"Using sentence transformer: {sentence_transformer.name}")

    llm = RagLLMChain(model=model)
    print("LLM created.")
    # sources = [
    #     "data/Victor_Hugo_Notre-Dame_De_Paris_en.epub",
    #     "data/Victor_Hugo_Les_Miserables_Fantine_1_of_5_en.epub",
    #     # "data/Victor_Hugo_Ruy_Blas_fr.epub",
    # ]
    # print("RAG populated.")
    # for source in sources:
    #     print(f"Loading {source}...")
    #     rag.add_document(source)
    # query = "\<human>\: " +  "Summarize this document : " + text + "\n" + "\<bot>\:"
    # previous_context: list[Document] = []
    # input_args = {
    #                 "context": {},
    #                 "query": query
    #             }
    # print("REQUEST", query)
    # response = llm.invoke(input_args)
    # input_args = text
    # print("INPUT_ARGS", input_args)
    # question = "What color is a banana?"
    response = llm.invoke({"text": text})
    # previous_request = response['query']
    # previous_context = response['context']
    # score = response.get('score')
    # if score is not None:
    #     print(f"Score: {response['score']}%. ", end="")
    # total_time = response['total_time']
    # llm_time = response['llm_time']
    # print(f"Total time: {total_time:.2f} seconds (LLM: {llm_time:.2f} s)")
    # print("TEXT", response['text'])
    return jsonify(
        {
            "response": response,
        }
    )


@app.get("/models")
@check_token()
async def list_models():
    """List all the configured models."""
    models = model_values.keys()
    return jsonify(
        {
            "models": list(models),
        }
    )
