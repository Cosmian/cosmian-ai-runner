# -*- coding: utf-8 -*-
"""
This module defines the Summarizer class for summarizing text using a Hugging Face model.
"""
from typing import Dict

from .model_pipeline import ModelPipeline


class Summarizer(ModelPipeline):
    """
    A class for summarizing text using a specified Hugging Face model.
    Attributes:
        model_name (str): The name of the model to be used for summarization.
        prefix (str): An optional prefix to use before the document to summarize.
        generation_config (Dict): A dictionary of configuration parameters for text generation.
    Methods:
        encode(text: str) -> torch.Tensor:
            Preprocess the input text and encode it into tokens.
        forward(input_tokens: torch.Tensor) -> torch.Tensor:
            Generate summary tokens from the input tokens.
        decode(output_tokens: torch.Tensor) -> str:
            Decode the summary tokens into a string.
    """

    def __init__(
        self,
        model_name: str,
        generation_config: Dict = {},
    ):
        super().__init__()

        self.model_name = model_name
        # optional prefix to use before the document to summarize
        self.prefix = generation_config.get("prefix", "")
        self.generation_config = {
            k: v for k, v in generation_config.items() if k != "prefix"
        }

    def encode(self, text):
        """Additional checks on the size of the input"""

        preprocess_text = self.prefix + text.strip().replace("\n", "")
        input_tokens = self.tokenizer.encode(
            preprocess_text, return_tensors="pt", truncation=True
        )
        if len(input_tokens[0]) < self.generation_config.get("min_length", 30):
            raise ValueError("Input text too short to summarize")

        return input_tokens

    def forward(self, input_tokens):
        return self.model.generate(input_tokens, **self.generation_config)[0]

    def decode(self, output_tokens):
        return self.tokenizer.decode(output_tokens, skip_special_tokens=True)
