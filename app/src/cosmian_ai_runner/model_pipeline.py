# -*- coding: utf-8 -*-
# Based on [Transformer Tools](https://github.com/huggingface/transformers/tree/main/src/transformers/tools)

from abc import ABC, abstractmethod
from typing import Any

import torch
from accelerate.utils import send_to_device
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class ModelPipeline(ABC):
    model_class = AutoModelForSeq2SeqLM
    tokenizer_class = AutoTokenizer
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Lazy init to save memory
    _model = None
    _tokenizer = None

    @abstractmethod
    def encode(self, text: str, *args, **kwargs) -> Any:
        """Check, pre-process and tokenize the input"""

    @abstractmethod
    def forward(self, input_tokens: Any) -> Any:
        """Run the model on preprocessed tokens"""

    @abstractmethod
    def decode(self, output_tokens: Any) -> str:
        """Decode and post-process the model output"""

    @property
    def model(self):
        if self._model is None:
            self._model = self.model_class.from_pretrained(self.model_name).to(
                self.device
            )
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self.tokenizer_class.from_pretrained(self.model_name)
        return self._tokenizer

    def __call__(self, *args, **kwargs) -> str:
        encoded_inputs = self.encode(*args, **kwargs)
        # move inputs to the same device as the model
        encoded_inputs = send_to_device(encoded_inputs, self.device)

        outputs = self.forward(encoded_inputs)
        # move the results back to CPU
        outputs = send_to_device(outputs, "cpu")

        decoded_outputs = self.decode(outputs)

        return decoded_outputs
