"""
Methods used for RAG
"""

from ebooklib import ITEM_DOCUMENT, epub
from bs4 import BeautifulSoup
from transformers import AutoTokenizer

from haystack import Pipeline, Document
from haystack.dataclasses import ByteStream
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)


def load_epub_as_bytes(epub_file_path):
    """
    To load an epub file - convert it as an HTML page and bytes
    """
    docs = []
    book = epub.read_epub(epub_file_path)

    # Find all paragraphs across sections
    for section in book.get_items_of_type(ITEM_DOCUMENT):
        section_html = section.get_body_content().decode("utf-8")
        section_soup = BeautifulSoup(section_html, "html.parser")
        paragraphs = section_soup.find_all("p")
        byte_stream: ByteStream
        for p in paragraphs:
            p_str = str(p)
            p_html = f"<html><head><title>Converted Epub</title></head><body>{p_str}</body></html>"
            # https://docs.haystack.deepset.ai/docs/data-classes#bytestream
            byte_stream = ByteStream(p_html.encode("utf-8"))
            docs.append(byte_stream)
    return docs


def load_document(converter, document_store, sources, reference):
    """
    Load document to the document_store, using a given converter and sources
    """
    doc_embedder = SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-MiniLM-L12-v2"
    )
    doc_embedder.warm_up()
    results = converter.run(sources=sources)
    converted_docs = results["documents"]
    converted_docs = [
        Document(content=doc.content, meta={"reference": reference})
        for doc in converted_docs
        if doc.content is not None
    ]
    converted_docs = list({doc.id: doc for doc in converted_docs}.values())

    cleaner = DocumentCleaner()
    cleaned_docs = cleaner.run(documents=converted_docs)["documents"]

    splitter = DocumentSplitter(split_by="sentence", split_length=2)
    split_docs = splitter.run(documents=cleaned_docs)["documents"]

    docs_with_embeddings = doc_embedder.run(split_docs)
    document_store.write_documents(docs_with_embeddings["documents"])


def chunk_text(text, model_name, max_tokens=256):
    """Split text into chunks of max_tokens."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.encode(text)
    chunks = [tokens[i : i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk) for chunk in chunks]


def build_rag_pipeline(retriever, generator):
    """
    Build rag pipeline
    """
    text_embedder = SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    text_embedder.warm_up()
    template = """
        Given the following information, answer the question.

        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{question}}
        Answer:
        """
    prompt_builder = PromptBuilder(template=template)
    pipeline = Pipeline()
    pipeline.add_component("text_embedder", text_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", generator)

    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "llm")

    return pipeline


def build_summarize_pipeline(model_name):
    """
    Build summarize pipeline
    """
    generator = HuggingFaceLocalGenerator(
        model_name,
        task="text2text-generation",
        generation_kwargs={
            "max_new_tokens": 100,
        },
    )
    generator.warm_up()
    pipeline = Pipeline()
    pipeline.add_component("summarizer", generator)
    return pipeline


def build_translate_pipeline(model_name):
    """
    Build translation pipeline
    """
    generator = HuggingFaceLocalGenerator(
        model_name,
        task="text2text-generation",
        generation_kwargs={
            "max_new_tokens": 500,
        },
    )
    generator.warm_up()
    pipeline = Pipeline()
    pipeline.add_component("translator", generator)

    return pipeline


def build_context_predict_pipeline(model_name):
    """
    Build context predict pipeline
    """
    generator = HuggingFaceLocalGenerator(
        model=model_name,
        task="text2text-generation",
    )
    generator.warm_up()
    template = """
        Given the following information, answer the question.

        Context: {{ context }}

        Question: {{ query }}?
        """
    pipeline = Pipeline()
    pipeline.add_component("prompt_builder", PromptBuilder(template=template))
    pipeline.add_component("llm", generator)
    pipeline.connect("prompt_builder", "llm")

    return pipeline


LANGUAGE_CODE = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "zh": "Chinese",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "nb": "Norwegian",
    "nn": "Norwegian",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sr": "Serbian",
    "es": "Spanish",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
}
