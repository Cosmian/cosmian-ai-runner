import nltk
import ssl

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
    from langchain_community.document_loaders import UnstructuredEPubLoader

    return UnstructuredEPubLoader(epub_path, mode="single").load()[0]
