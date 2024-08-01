# -*- coding: utf-8 -*-
import os
import tempfile
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
from .vector_db import STValue

torch.set_num_threads(os.cpu_count() or 1)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024 * 1024  # 1 GB
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
if len(data_list) > 0:
    for item in data_list:
        model_id = item.get("model_id")
        file = item.get("file")
        prompt = item.get("prompt")
        task = item.get("task")
        kwargs = item.get("kwargs")
        model_value = ModelValue(model_id, file, prompt, task, kwargs)
        model_values[model_id] = model_value

element = AppConfig.get_sentence_transformer_config()
if element:
    file = element.get("file")
    score_threshold = element.get("score_threshold")
    sentence_transformer = STValue(file, score_threshold)

    print(f"Using sentence transformer: {sentence_transformer.file}")

    rag = Rag(sentence_transformer=sentence_transformer)
    print("RAG created.")
else:
    rag = None
    print("No sentence tranformer configured, RAG is not created.")

# sources = [
#     "data/Victor_Hugo_Notre-Dame_De_Paris_en.epub",
#     # "data/Victor_Hugo_Les_Miserables_Fantine_1_of_5_en.epub",
#     # "data/Victor_Hugo_Ruy_Blas_fr.epub",
# ]
# for source in sources:
#     print(f"Loading {source}...")
#     rag.add_document(source)
# print("RAG populated.")


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
    if rag is None:
        return ("RAG is not created - fobidden request", 404)
    if "text" not in request.form:
        return ("Error: Missing text content", 400)

    query = request.form["text"]
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
        response = rag.invoke(model=model, query=query)
        return jsonify(
            {
                "response": response,
            }
        )
    except Exception as e:
        return (f"Error during prediction: {e}", 400)


@app.post("/add_document")
@check_token()
async def add_doc():
    if rag is None:
        return ("RAG is not created - fobidden request", 404)
    if not os.path.exists("data"):
        os.makedirs("data")

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith(".epub"):
        file_ext = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(
            dir="data", suffix=file_ext, delete=False
        ) as temp_file:
            temp_file_name = temp_file.name
            file.save(temp_file_name)

        try:
            rag.add_document(temp_file_name)
            return ("File successfully processed", 200)
        except Exception as e:
            return (f"Error adding file: {e}", 400)
        finally:
            os.remove(temp_file_name)
    else:
        return ("Invalid file extension - must be epub.", 400)
