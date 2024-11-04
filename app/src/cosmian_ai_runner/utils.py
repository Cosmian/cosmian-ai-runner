"""
Methods used for RAG
"""
from ebooklib import ITEM_DOCUMENT, epub
from bs4 import BeautifulSoup
from transformers import AutoTokenizer

from haystack import Pipeline, Document
from haystack.dataclasses import ByteStream
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.builders import PromptBuilder


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


def load_document(converter, document_store, doc_embedder, sources, reference):
    """
    Load document to the document_store, using a given converter and sources
    """
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


def build_rag_pipeline(text_embedder, retriever, template, generator):
    """
    Build rag pipeline
    """
    prompt_builder = PromptBuilder(template=template)
    rag_pipeline = Pipeline()
    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("llm", generator)

    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")

    return rag_pipeline


def chunk_text(text, model_name, max_tokens=256):
    """Split text into chunks of max_tokens."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk) for chunk in chunks]
