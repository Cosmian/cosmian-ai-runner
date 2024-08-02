# -*- coding: utf-8 -*-
"""
This module provides functionality for verifying JWT tokens in a Flask application.
It includes a function to verify tokens against OpenID Connect configurations and
a decorator to protect routes by requiring a valid JWT token.
"""
from functools import wraps
from typing import Any, Dict, List

import jwt
from flask import request

from .config import AppConfig

PREFIX = "Bearer "


def verify_token(id_token: str, openid_configs: List[Dict]) -> Any:
    """
    Verify a JWT token against a list of OpenID Connect configurations.

    Args:
        id_token (str): The JWT token to be verified.
        openid_configs (List[Dict]): A list of OpenID Connect configurations, each containing
                                     a 'jwks_uri' and 'client_id'.

    Returns:
        Any: The decoded token if verification is successful.

    Raises:
        jwt.PyJWKClientError: If no matching signing key is found or if token verification fails.
    """
    header = jwt.get_unverified_header(id_token)
    for conf in openid_configs:
        jwks_client = jwt.PyJWKClient(conf["jwks_uri"])

        try:
            # try matching the user token with the current jwks
            signing_key = jwks_client.get_signing_key(header["kid"])
        except Exception:
            # try next jwks
            continue
        else:
            # decode and verify user token
            data = jwt.decode(
                id_token,
                key=signing_key.key,
                algorithms=header["alg"],
                audience=conf["client_id"],
            )
            # return decoded token
            return data

    raise jwt.PyJWKClientError("Invalid token: Unable to find a matching signing key")


def check_token():
    """
    Decorator to protect routes by requiring a valid JWT token.

    This decorator checks for the presence of an 'Authorization' header with a Bearer token.
    It verifies the token using the OpenID Connect configurations specified in the application's
    authentication configuration.

    Returns:
        function: The decorated function with token verification.
    """
    def decorator(f):
        auth_config = AppConfig.get_auth_config()
        if not auth_config:
            return f

        @wraps(f)
        async def wrapper(*args, **kwargs):
            if "Authorization" in request.headers:
                bearer_token = request.headers["Authorization"]
                if not bearer_token.startswith(PREFIX):
                    return ("Error: Bearer not found", 401)
                prefix_len = len(PREFIX)
                id_token = bearer_token[prefix_len:]
                try:
                    _ = verify_token(id_token, auth_config["openid_configs"])
                    return await f(*args, **kwargs)
                except Exception as e:
                    return (f"Error: {e}", 401)

            return ("Error: Missing bearer token", 401)

        return wrapper

    return decorator
