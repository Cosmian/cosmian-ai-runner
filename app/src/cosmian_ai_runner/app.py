# -*- coding: utf-8 -*-
"""
This module defines the Flask web application for text summarization, translation,
retrieval-augmented generation (RAG), and model predictions using Hugging Face models.
"""
import os
import tempfile
from http import HTTPStatus

import torch
from asgiref.wsgi import WsgiToAsgi
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from haystack import Pipeline
from haystack.utils import Secret
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.converters import (
    HTMLToDocument,
    DOCXToDocument,
    PyPDFToDocument
)
# from fastrag.generators.openvino import OpenVINOGenerator

from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore


from .auth import check_token
from .config import AppConfig
from .utils import load_document, load_epub_as_bytes, build_rag_pipeline, chunk_text
torch.set_num_threads(os.cpu_count() or 1)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024 * 1024  # 1 GB
app_asgi = WsgiToAsgi(app)
CORS(app, resources={r"/*": {"origins": "*"}})

config_path = os.getenv("CONFIG_PATH", "config.json")
with open(config_path, encoding="utf-8") as f:
    AppConfig.load(f)

# Prepare elements for each documentary database
template = """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""
doc_embedder = SentenceTransformersDocumentEmbedder(
    model="sentence-transformers/all-MiniLM-L12-v2"
)
doc_embedder.warm_up()

# Litterature basis
document_store_litterature = ChromaDocumentStore(
    persist_path="rag_epub_doc_store_litterature"
)
text_embedder_litterature = SentenceTransformersTextEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)
retriever_litterature = ChromaEmbeddingRetriever(document_store_litterature)

# openvino_compressed_model_path = "./model"
generator_litterature = HuggingFaceLocalGenerator(
    model="google/flan-t5-large",
    task="text2text-generation",
    token=Secret.from_env_var("HF_API_TOKEN"),
)
# generator_litterature = OpenVINOGenerator(
#     model="google/flan-t5-xl",
#     compressed_model_dir=openvino_compressed_model_path,
#     device_openvino="CPU",
#     task="text-generation",
#     generation_kwargs={
#         "max_new_tokens": 100,
#     }
# )
pipeline_litterature = build_rag_pipeline(
    text_embedder_litterature,
    retriever_litterature,
    template,
    generator_litterature,
)

# Science basis
document_store_science = ChromaDocumentStore(
    persist_path="rag_epub_doc_store_science"
)
text_embedder_science = SentenceTransformersTextEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)
retriever_science = ChromaEmbeddingRetriever(document_store_science)
generator_science = HuggingFaceLocalGenerator(
    model="google/flan-t5-large",
    task="text2text-generation",
    token=Secret.from_env_var("HF_API_TOKEN"),
)
pipeline_science = build_rag_pipeline(
    text_embedder_science,
    retriever_science,
    template,
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

    try:
        generator = HuggingFaceLocalGenerator(
            model_name,
            task="text2text-generation",
            generation_kwargs={
                "num_beams": 1,
                "early_stopping": True,
                "num_return_sequences": 1,
                "max_length": 512,
            },
        )
        generator.warm_up()
        pipeline = Pipeline()
        pipeline.add_component("summarizer", generator)
        chunks = chunk_text(text, model_name)
        all_replies = []
        for chunk in chunks:
            data = {"summarizer": {"prompt": f"Summarize this text: {chunk}"}}
            result = pipeline.run(data=data)["summarizer"]["replies"][0]
            all_replies.append(result)
        combined_result = " ".join(all_replies)
        data = {"summarizer": {"prompt": f"Summarize this text: {combined_result}"}}
        summary = pipeline.run(data=data)["summarizer"]["replies"]
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

    try:
        generator = HuggingFaceLocalGenerator(
            model_name,
            task="text2text-generation",
            token=Secret.from_env_var("HF_API_TOKEN"),
        )
        generator.warm_up()
        pipeline = Pipeline()
        pipeline.add_component("translater", generator)
        chunks = chunk_text(text, model_name)
        all_replies = []
        for chunk in chunks:
            data = {"translater": {"prompt": f"{chunk}"}}
            result = pipeline.run(data=data)["translater"]["replies"][0]
            all_replies.append(result)
        combined_result = " ".join(all_replies)
    except OSError as e:
        return (f"Error: model does not support these languages for translation.", 400)
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

    try:
        generator = HuggingFaceLocalGenerator(
            model="google/flan-t5-large",
            task="text2text-generation",
            generation_kwargs={
                "max_new_tokens": 100,
                "temperature": 0.9,
                "max_length": 512,
            },
        )
        template = """
            Given the following information, answer the question.

            Context: {{ context }}

            Question: {{ query }}?
            """
        pipeline = Pipeline()
        pipeline.add_component("prompt_builder", PromptBuilder(template=template))
        pipeline.add_component("llm", generator)
        pipeline.connect("prompt_builder", "llm")
        data = {"prompt_builder": {"query": query, "context": context}}
        result = pipeline.run(data=data)["llm"]["replies"]

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

    try:
        pipeline = database["pipeline"]
        response = pipeline.run(
            {
                "text_embedder": {"text": query},
                "prompt_builder": {"question": query},
            }
        )
        return jsonify(
            {
                "result": response["llm"]["replies"],
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

    database = next(
        (obj for obj in documentary_bases if obj["name"] == database_name), None
    )
    if database is None:
        return ("Error database not found", 404)

    if file.filename == "":
        return jsonify({"Error": "No selected file"}), 400

    if file and (
        file.filename.endswith(".epub")
        or file.filename.endswith(".docx")
        or file.filename.endswith(".pdf")
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
            load_document(
                converter=converter,
                document_store=database["document_store"],
                doc_embedder=doc_embedder,
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
