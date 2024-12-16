# -*- coding: utf-8 -*-
"""
This module defines the Flask web application for text summarization, translation,
retrieval-augmented generation (RAG), and model predictions using Hugging Face models.
"""
import os
import tempfile
from http import HTTPStatus
from contextlib import nullcontext

from asgiref.wsgi import WsgiToAsgi
import torch
from flask import Flask, Response, jsonify, request, current_app
from flask_cors import CORS

from haystack.utils import Secret
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.converters import (
    HTMLToDocument,
    DOCXToDocument,
    PyPDFToDocument,
)
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from .auth import check_token
from .config import AppConfig
from .utils import (
    load_document,
    load_epub_as_bytes,
    build_rag_pipeline,
    chunk_text,
    build_summarize_pipeline,
    build_context_predict_pipeline,
    build_translate_pipeline,
)

torch.set_num_threads(os.cpu_count() or 1)

app = Flask(__name__)
app_asgi = WsgiToAsgi(app)


def create_app():
    """
    Setup device and autocast context, checking the hardware configuration and the amx option
    """
    app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024 * 1024  # 1 GB
    CORS(app, resources={r"/*": {"origins": "*"}})

    config_path = os.getenv("CONFIG_PATH", "config.json")
    with open(config_path, encoding="utf-8") as f:
        AppConfig.load(f)

    # Determine the device
    # Read the AMX flag from the environment variable
    amx_enabled = os.getenv("AMX_ENABLED", "0") == "1"

    if amx_enabled:
        print("AMX extension is enabled!")
    else:
        print("AMX extension is disabled.")

    if torch.cuda.is_available():
        print("Application is using GPU")
        torch.device("cuda")  # Use GPU if available
        autocast_context = torch.amp.autocast("cuda")
    elif torch.backends.cpu and amx_enabled:  # Check for Intel AMX
        print("Application is using AMX")
        torch.device("cpu")
        autocast_context = torch.amp.autocast("cpu")
    elif torch.backends.mps.is_available():  # Check for Metal on macOS
        print("Application is using MPS")
        torch.device("mps")
        autocast_context = None  # Metal backend does not support autocast
    else:
        print("Application is using CPU")
        torch.device("cpu")
        autocast_context = None  # Default CPU without autocast

    # Attach context to the app object
    app.config["AUTOMATIC_CAST_CONTEXT"] = autocast_context or nullcontext()


# Build documentary database

# Litterature basis
document_store_litterature = ChromaDocumentStore(
    persist_path="rag_epub_doc_store_litterature"
)
retriever_litterature = ChromaEmbeddingRetriever(document_store_litterature)
generator_litterature = HuggingFaceLocalGenerator(
    model="google/flan-t5-large",
    task="text2text-generation",
    token=Secret.from_env_var("HF_API_TOKEN"),
    generation_kwargs={
        "max_new_tokens": 500,
    },
)
pipeline_litterature = build_rag_pipeline(
    retriever_litterature,
    generator_litterature,
)

# Science basis
document_store_science = ChromaDocumentStore(persist_path="rag_epub_doc_store_science")
retriever_science = ChromaEmbeddingRetriever(document_store_science)
generator_science = HuggingFaceLocalGenerator(
    model="google/flan-t5-large",
    task="text2text-generation",
    token=Secret.from_env_var("HF_API_TOKEN"),
    generation_kwargs={
        "max_new_tokens": 500,
    },
)
pipeline_science = build_rag_pipeline(
    retriever_science,
    generator_science,
)

# Documentary bases
documentary_bases = [
    {
        "name": "litterature",
        "document_store": document_store_litterature,
        "pipeline": pipeline_litterature,
        "references": [],
    },
    {
        "name": "science",
        "document_store": document_store_science,
        "pipeline": pipeline_science,
        "references": [],
    },
]


@app.get("/health")
def health_check():
    """Health check of the application."""
    return Response(response="OK", status=HTTPStatus.OK)


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
    model_name = "facebook/bart-large-cnn"
    autocast_context = current_app.config["AUTOMATIC_CAST_CONTEXT"]

    try:
        with autocast_context:
            summarize_pipeline = build_summarize_pipeline(model_name)
            chunks = chunk_text(text, model_name, 800)
            all_replies = []
            for chunk in chunks:
                data = {"summarizer": {"prompt": f"Summarize this text: {chunk}"}}
                result = summarize_pipeline.run(data=data)["summarizer"]["replies"][0]
                all_replies.append(result)
            combined_result = " ".join(all_replies)
            data = {"summarizer": {"prompt": f"Summarize this text: {combined_result}"}}
            summary = summarize_pipeline.run(data=data)["summarizer"]["replies"][0]
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

    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    autocast_context = current_app.config["AUTOMATIC_CAST_CONTEXT"]

    try:
        with autocast_context:
            pipeline = build_translate_pipeline(model_name)
            all_replies = []
            chunks = chunk_text(text, model_name, 200)
            for chunk in chunks:
                data = {"translater": {"prompt": f"{chunk}"}}
                result = pipeline.run(data=data)["translater"]["replies"][0]
                all_replies.append(result)
            combined_result = " ".join(all_replies)
    except OSError:
        return ("Error: model does not support these languages for translation.", 400)
    except ValueError as e:
        return (f"Error: {e}", 400)

    return jsonify(
        {
            "translation": combined_result,
        }
    )


@app.post("/context_predict")
@check_token()
async def context_predict():
    """
    Get prediction using current text as context.
    Expects 'doc' in form data.
    Returns a JSON response with the answer.
    """
    if "query" not in request.form:
        return ("Error: Missing query", 400)
    if "context" not in request.form:
        return ("Error: Missing text content", 400)

    query = request.form["query"]
    context = request.form["context"]
    model_name = "google/flan-t5-large"
    pipeline_context_predict = build_context_predict_pipeline(model_name)
    autocast_context = current_app.config["AUTOMATIC_CAST_CONTEXT"]

    try:
        with autocast_context:
            data = {"prompt_builder": {"query": query, "context": context}}
            result = pipeline_context_predict.run(data=data)["llm"]["replies"]
    except ValueError as e:
        return (str(e), 400)

    return jsonify(
        {
            "result": result,
        }
    )


@app.get("/documentary_bases")
@check_token()
async def list_databases():
    """
    List all the configured databases (model, associated sentence transformer and
    stored references).
    Returns a JSON response with the list of RAG.
    """
    databases = {}
    for db in documentary_bases:
        name = db["name"]
        references = db["references"]
        databases[name] = references
    return jsonify(
        {
            "documentary_bases": databases,
        }
    )


@app.post("/rag_predict")
@check_token()
async def query_rag():
    """
    Make a prediction using the loaded RAG.
    Expects 'text' and 'database' in form data.
    Returns a JSON response with the RAG prediction and its context.
    """
    if "query" not in request.form:
        return ("Error: Missing query content", 400)
    if "db" not in request.form:
        return ("Error: Missing selected documentary basis", 400)

    query = request.form["query"]
    database_name = request.form["db"]

    database = next(
        (obj for obj in documentary_bases if obj["name"] == database_name), None
    )
    if database is None:
        return ("Error database not found", 404)
    autocast_context = current_app.config["AUTOMATIC_CAST_CONTEXT"]

    try:
        with autocast_context:
            pipeline = database["pipeline"]
            response = pipeline.run(
                {
                    "text_embedder": {"text": query},
                    "prompt_builder": {"question": query},
                }
            )["llm"]["replies"]
            return jsonify(
                {
                    "result": response,
                }
            )
    except Exception as e:
        return (f"Error during prediction {e}", 400)


@app.post("/add_reference")
@check_token()
async def add_ref():
    """
    Load a reference into the vector database.
    Expects a file with '.epub' extension in form data.
    Returns a success message or error.
    """
    if "db" not in request.form:
        return ("Error: Missing selected documentary basis", 400)
    if "reference" not in request.form:
        return ("Error: Missing reference", 400)
    if not os.path.exists("data"):
        os.makedirs("data")
    if "file" not in request.files:
        return jsonify({"Error": "No file part"}), 400

    file = request.files["file"]
    database_name = request.form["db"]
    reference = request.form["reference"]
    autocast_context = current_app.config["AUTOMATIC_CAST_CONTEXT"]

    database = next(
        (obj for obj in documentary_bases if obj["name"] == database_name), None
    )
    if database is None:
        return ("Error database not found", 404)

    if file.filename == "":
        return jsonify({"Error": "No selected file"}), 400

    if file and (
        file.filename.endswith(".epub") or file.filename.endswith(".docx") or file.filename.endswith(".pdf")
    ):
        file_ext = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(
            dir="data", suffix=file_ext, delete=False
        ) as temp_file:
            temp_file_name = temp_file.name
            file.save(temp_file_name)

        try:
            if file.filename.endswith(".epub"):
                converter = HTMLToDocument()
                sources = load_epub_as_bytes(temp_file_name)
            elif file.filename.endswith(".docx"):
                converter = DOCXToDocument()
                sources = [temp_file_name]
            else:
                converter = PyPDFToDocument()
                sources = [temp_file_name]
            with autocast_context:
                load_document(
                    converter=converter,
                    document_store=database["document_store"],
                    sources=sources,
                    reference=reference,
                )
            database["references"].append(reference)
            return ("File successfully processed", 200)
        except Exception as e:
            return (f"Error adding file: {e}", 400)
        finally:
            os.remove(temp_file_name)
    else:
        return (
            {"Error": "Invalid file extension - must be epub, docx or PDF."},
            400,
        )


@app.delete("/delete_reference")
@check_token()
async def delete_ref():
    """
    Delete a reference into the vector database.
    Returns a success message or error.
    """
    if "db" not in request.form:
        return ("Error: Missing selected documentary basis", 400)
    if "reference" not in request.form:
        return ("Error: Missing reference", 400)

    database_name = request.form["db"]
    reference = request.form["reference"]

    database = next(
        (obj for obj in documentary_bases if obj["name"] == database_name), None
    )
    if database is None:
        return ("Error database not found", 404)

    try:
        document_store = database["document_store"]
        documents = document_store.filter_documents(
            filters={"field": "reference", "operator": "==", "value": reference}
        )
        doc_ids = [doc.id for doc in documents]
        database["references"].remove(reference)
        document_store.delete_documents(doc_ids)
        return ("Reference successfully removed", 200)
    except Exception as e:
        return (f"Error removing reference: {e}", 400)
