# -*- coding: utf-8 -*-
import json
from typing import Dict, Optional, TextIO


class AppConfig:
    _config: Optional[Dict] = None

    @classmethod
    def load(cls, config_file: TextIO) -> Dict:
        cls._config = json.load(config_file)
        return cls.get()

    @classmethod
    def get(cls) -> Dict:
        if cls._config:
            return cls._config
        raise AttributeError("Missing or empty config")

    @classmethod
    def get_auth_config(cls) -> Optional[Dict]:
        config = cls.get()
        if "auth" in config:
            return config["auth"]
        return None

    @classmethod
    def get_models_config(cls) -> Optional[list]:
        config = cls.get()
        if "models" in config:
            return config["models"]
        return None
