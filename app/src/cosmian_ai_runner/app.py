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
from .summarizer import Summarizer
from .translator import Translator

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
