# -*- coding: utf-8 -*-
"""
This module defines the Flask web application for text summarization, translation,
retrieval-augmented generation (RAG), and model predictions using Hugging Face models.
"""
import os
import tempfile
from http import HTTPStatus
from typing import Dict

import torch
from asgiref.wsgi import WsgiToAsgi
from flask import Flask, Response, jsonify, request, current_app
from flask_cors import CORS

from .auth import check_token
from .config import AppConfig
from .detect import is_gpu_available
from .llm_chain import  ModelValue
from .rag import Rag
from .summarizer import Summarizer
from .translator import Translator
from .vector_db import STValue

torch.set_num_threads(os.cpu_count() or 1)

def create_app():
    app = Flask(__name__)

    app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024 * 1024  # 1 GB
    app_asgi = WsgiToAsgi(app)
    CORS(app, resources={r"/*": {"origins": "*"}})

    config_path = os.getenv("CONFIG_PATH", "config.json")
    with open(config_path, encoding='utf-8') as f:
        AppConfig.load(f)
        summary_config = AppConfig.get_summary_config()
        translation_config = AppConfig.get_translation_config()
        databases_config = AppConfig.get_databases_config()

    with app.app_context():
        summarizer_by_lang: Dict[str, Summarizer] = {
            lang: Summarizer(**model_config)
            for lang, model_config in summary_config.items()
        }
        current_app.summarizer_by_lang = summarizer_by_lang

        translator = Translator(**translation_config)
        current_app.translator = translator

        database_values = {}
        data_list = databases_config
        if data_list is not None and len(data_list) > 0:
            for item in data_list:
                database_name = item.get("name")
                model_config = item.get("model")
                model_id = model_config.get("model_id")
                file = model_config.get("file")
                prompt = model_config.get("prompt")
                task = model_config.get("task")
                kwargs = model_config.get("kwargs")
                model_value = ModelValue(model_id, file, prompt, task, kwargs)

                sentence_transformer_config = item.get("sentence_transformer")
                file = sentence_transformer_config.get("file")
                score_threshold = sentence_transformer_config.get("score_threshold")
                sentence_transformer = STValue(file, score_threshold)
                rag = Rag(sentence_transformer=sentence_transformer)

                database_values[database_name] = {
                    "model" : model_value,
                    "rag": rag,
                    "references": []
                }
        current_app.database_values = database_values

    @app.post("/summarize")
    @check_token()
    async def post_summarize():
        """
        Summarize text.
        Expects 'doc' and optionally 'src_lang' in form data.
        Returns a JSON response with the summary.
        """
        if "doc" not in request.form:
            return ("Error: Missing file content", 400)

        text = request.form["doc"]
        src_lang = request.form.get("src_lang", default="default")

        try:
            summarizer = current_app.summarizer_by_lang.get(src_lang, current_app.summarizer_by_lang["default"])
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
        """
        Translate text.
        Expects 'doc', 'src_lang', and 'tgt_lang' in form data.
        Returns a JSON response with the translation.
        """
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
            result = current_app.translator(text, src_lang=src_lang, tgt_lang=tgt_lang)
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


    @app.get("/databases")
    @check_token()
    async def list_databases():
        """
        List all the configured databases (model, associated sentence transformer and
        stored references).
        Returns a JSON response with the list of RAG.
        """
        names = list(current_app.database_values.keys())
        databases = {}
        for name in names:
            references = current_app.database_values[name]['references']
            databases[name] = references
        return jsonify(
            {
                "databases": databases,
            }
        )


    @app.post("/predict")
    @check_token()
    async def build_rag():
        """
        Make a prediction using the loaded RAG.
        Expects 'text' and 'database' in form data.
        Returns a JSON response with the RAG prediction and its context.
        """
        if "text" not in request.form:
            return ("Error: Missing text content", 400)
        if "database" not in request.form:
            return ("Error: Missing selected database", 400)

        query = request.form["text"]
        database_name = request.form["database"]

        if is_gpu_available():
            print("GPU is available.")

        try:
            database = current_app.database_values[database_name]
        except KeyError as e:
            return (f"Error database not found: {e}", 404)

        try:
            model = database["model"]
            rag = database["rag"]
        except KeyError as e:
            return (f"Error model/RAG not found: {e}", 404)

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


    @app.post("/add_reference")
    @check_token()
    async def add_ref():
        """
        Load a reference into the vector database.
        Expects a file with '.epub' extension in form data.
        Returns a success message or error.
        """
        if "database" not in request.form:
            return ("Error: Missing selected database", 400)
        if "reference" not in request.form:
            return ("Error: Missing reference", 400)
        if not os.path.exists("data"):
            os.makedirs("data")
        if "file" not in request.files:
            return jsonify({"Error": "No file part"}), 400

        file = request.files["file"]
        database_name = request.form["database"]
        reference = request.form["reference"]

        try:
            database = current_app.database_values[database_name]
        except KeyError as e:
            return (f"Error database not found: {e}", 404)

        if file.filename == "":
            return jsonify({"Error": "No selected file"}), 400

        if file and file.filename.endswith(".epub") and database_name:
            file_ext = os.path.splitext(file.filename)[1]
            with tempfile.NamedTemporaryFile(
                dir="data", suffix=file_ext, delete=False
            ) as temp_file:
                temp_file_name = temp_file.name
                file.save(temp_file_name)

            try:
                rag = database["rag"]
            except KeyError as e:
                return (f"Error model/RAG not found: {e}", 404)

            try:
                rag.add_document(temp_file_name, reference)
                database["references"].append(reference)
                return ("File successfully processed", 200)
            except Exception as e:
                return (f"Error adding file: {e}", 400)
            finally:
                os.remove(temp_file_name)
        else:
            return ({"Error": "Invalid file extension - must be epub."}, 400)


    @app.delete("/delete_reference")
    @check_token()
    async def delete_ref():
        """
        Delete a reference into the vector database.
        Returns a success message or error.
        """
        if "database" not in request.form:
            return ("Error: Missing selected database", 400)
        if "reference" not in request.form:
            return ("Error: Missing reference", 400)

        database_name = request.form["database"]
        reference = request.form["reference"]

        try:
            database = current_app.database_values[database_name]
        except KeyError as e:
            return (f"Error database not found: {e}", 404)

        try:
            rag = database["rag"]
        except KeyError as e:
            return (f"Error model/RAG not found: {e}", 404)

        try:
            if reference in database["references"]:
                rag.delete_reference(reference)
                database["references"].remove(reference)
                return ("Reference successfully removed", 200)
            return ("Error: Reference not found from specified database", 404)
        except Exception as e:
            return (f"Error removing reference: {e}", 400)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
