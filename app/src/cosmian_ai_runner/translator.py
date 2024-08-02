# -*- coding: utf-8 -*-
"""
This module defines the Translator class for translating text using a Hugging Face model.
"""
from typing import Dict, List

import torch
from transformers import PreTrainedTokenizer

from .model_pipeline import ModelPipeline

# ISO 639 language codes based on `https://en.wikipedia.org/wiki/Languages_used_on_the_Internet`
LANGUAGE_CODES = {
    "ar": "arb_Arab",
    "bg": "bul_Cyrl",
    "ca": "cat_Latn",
    "cs": "ces_Latn",
    "da": "dan_Latn",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "et": "est_Latn",
    "fa": "pes_Arab",
    "fi": "fin_Latn",
    "fr": "fra_Latn",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "hr": "hrv_Latn",
    "hu": "hun_Latn",
    "id": "ind_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    "nb": "nno_Latn",
    "nl": "nld_Latn",
    "nn": "nob_Latn",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "sr": "srp_Cyrl",
    "sv": "swe_Latn",
    "th": "tha_Thai",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "vi": "vie_Latn",
    "zh": "zho_Hans",
}


class Translator(ModelPipeline):
    """
    A class for translating text using a specified Hugging Face model.
    Attributes:
        model_name (str): The name of the model to be used for translation.
        generation_config (Dict): A dictionary of configuration parameters for text generation.
        lang_to_code (Dict): A dictionary mapping language codes to the corresponding language format.
    Methods:
        encode(text: str, src_lang: str, tgt_lang: str) -> Dict[str, torch.Tensor]:
            Preprocess the input text and encode it into tokens for the source and target languages.
        forward(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
            Generate translation tokens from the input tokens.
        decode(outputs: torch.Tensor) -> str:
            Decode the translation tokens into a string.
    """
    def __init__(self, model_name: str, generation_config: Dict = {}):
        self.generation_config = generation_config
        self.lang_to_code = LANGUAGE_CODES
        self.model_name = model_name

    def encode(self, text: str, src_lang, tgt_lang):
        if src_lang not in self.lang_to_code:
            raise ValueError(f"{src_lang} is not a supported language.")
        if tgt_lang not in self.lang_to_code:
            raise ValueError(f"{tgt_lang} is not a supported language.")
        src_lang = self.lang_to_code[src_lang]
        tgt_lang = self.lang_to_code[tgt_lang]

        inputs = self.tokenizer._build_translation_inputs(
            text, return_tensors="pt", src_lang=src_lang, tgt_lang=tgt_lang
        )

        # max length to get a meaningful output, we will split our input in chunks accordingly
        chunk_size = int(0.75 * self.generation_config.get("max_length", 200))
        input_tokens = inputs["input_ids"][0].tolist()
        chunks = split_chunks(input_tokens, chunk_size, self.tokenizer)

        return {
            "input_ids": torch.tensor(chunks),
            "forced_bos_token_id": inputs["forced_bos_token_id"],
        }

    def forward(self, inputs):
        return self.model.generate(
            **inputs,
            **self.generation_config,
        )

    def decode(self, outputs):
        decoded_batch = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return "\n".join(decoded_batch)


def find_separator_index(input_tokens, sep_token, start_index):
    """
    Find the index of the separator token closest to the specified start index.
    Args:
        input_tokens (List[int]): The list of input tokens.
        sep_token (int): The separator token ID.
        start_index (int): The start index for searching the separator token.
    Returns:
        int: The index of the separator token or the start index if not found.
    """
    split_index = start_index
    while input_tokens[split_index] != sep_token:
        split_index -= 1
        if split_index == 0:
            return start_index
    return split_index


def end_pad_tokens(input_tokens, length, pad_token, eos_token=None):
    """
    Pad the input tokens to the specified length, optionally adding an EOS token.
    Args:
        input_tokens (List[int]): The list of input tokens.
        length (int): The desired length of the padded tokens.
        pad_token (int): The pad token ID.
        eos_token (int, optional): The EOS token ID. Defaults to None.
    Returns:
        List[int]: The padded list of tokens.
    """
    if eos_token:
        input_tokens.append(eos_token)
    input_tokens.extend([pad_token] * (length - len(input_tokens)))
    assert len(input_tokens) == length
    return input_tokens


def split_chunks(
    input_tokens: List[int], max_tokens_length: int, tokenizer: PreTrainedTokenizer
):
    """Cut the input tokens in multiple chunks of `max_tokens_length`
    This methods tries to find a `separator token` (like '.') to avoid cutting
    at the middle of a phrase which would lead to poor translation.

    Returns:
        one or more padded list of tokens
    """
    if len(input_tokens) <= max_tokens_length:
        return [input_tokens]

    sep_token = tokenizer.encode(".", add_special_tokens=False)[0]
    bos_token = input_tokens[0]
    eos_token = input_tokens[-1]

    chunks = []
    # Split at the previous `sep` closer to `max_tokens_length`
    while len(input_tokens) > max_tokens_length:
        split_index = find_separator_index(
            input_tokens, sep_token, max_tokens_length - 1
        )
        chunks.append(
            end_pad_tokens(
                input_tokens[:split_index],
                max_tokens_length,
                tokenizer.pad_token_id,
                eos_token,
            )
        )
        input_tokens = input_tokens[split_index:]
        input_tokens[0] = bos_token

    # Add last chunk
    chunks.append(
        end_pad_tokens(
            input_tokens,
            max_tokens_length,
            tokenizer.pad_token_id,
        )
    )
    return chunks
