# -*- coding: utf-8 -*-
"""
This module provides the AppConfig class for managing application configuration.
It includes methods for loading the configuration from a file and retrieving
specific sections of the configuration.
"""
import json
from typing import Dict, Optional, TextIO


class AppConfig:
    """
    A class to manage application configuration.
    This class provides methods to load configuration from a file and retrieve
    specific sections of the configuration such as authentication, models, and
    sentence transformer configurations.
    """
    _config: Optional[Dict] = None

    @classmethod
    def load(cls, config_file: TextIO) -> Dict:
        """
        Load configuration from a JSON file.
        Args:
            config_file (TextIO): The configuration file object to load from.
        Returns:
            Dict: The loaded configuration.
        Raises:
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        cls._config = json.load(config_file)
        return cls.get()

    @classmethod
    def get(cls) -> Dict:
        """
        Get the loaded configuration.
        Returns:
            Dict: The loaded configuration.
        Raises:
            AttributeError: If the configuration is missing or empty.
        """
        if cls._config:
            return cls._config
        raise AttributeError("Missing or empty config")

    @classmethod
    def get_auth_config(cls) -> Optional[Dict]:
        """
        Get the authentication configuration.
        Returns:
            Optional[Dict]: The authentication configuration if available, otherwise None.
        """
        config = cls.get()
        if "auth" in config:
            return config["auth"]
        return None

    @classmethod
    def get_models_config(cls) -> Optional[list]:
        """
        Get the models configuration.
        Returns:
            Optional[list]: The models configuration if available, otherwise None.
        """
        config = cls.get()
        if "models" in config:
            return config["models"]
        return None

    @classmethod
    def get_sentence_transformer_config(cls) -> Optional[list]:
        """
        Get the sentence transformer configuration.
        Returns:
            Optional[list]: The sentence transformer configuration if available, otherwise None.
        """
        config = cls.get()
        if "sentence_transformer" in config:
            return config["sentence_transformer"]
        return None
