# -*- coding: utf-8 -*-
"""Main function"""
import argparse
import asyncio

from hypercorn.asyncio import serve
from hypercorn.config import Config

from .app import app_asgi, create_app


def main():
    """
    Main function to parse command-line arguments and start the Hypercorn server.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--port", type=int, default=5000, help="The listening port"
    )
    args = parser.parse_args()
    config_map = {
        "bind": f"0.0.0.0:{args.port}",
        "alpn_protocols": ["h2"],
        "workers": 1,
        "accesslog": "-",
        "errorlog": "-",
        "worker_class": "uvloop",
        "wsgi_max_body_size": 2 * 1024 * 1024 * 1024,  # 2 GB
    }

    config = Config.from_mapping(config_map)
    create_app()
    asyncio.run(serve(app_asgi, config))


if __name__ == "__main__":
    main()
