"""Util to load epub file"""

import ssl

import nltk
from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain_core.documents import Document

# This is a workaround for the SSL certificate issue that pops from time to time
# when downloading the punkt tokenizer from nltk
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download("punkt")


def load_epub(epub_path: str) -> Document:
    """
    Load an EPUB file and return it as a Document.
    Args:
        epub_path (str): Path to the EPUB file.
    Returns:
        Document: The loaded EPUB content wrapped in a Document object.
    """
    return UnstructuredEPubLoader(epub_path, mode="single").load()[0]
