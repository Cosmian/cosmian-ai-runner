# -*- coding: utf-8 -*-
import argparse
import asyncio
import json
from pathlib import Path

import requests

cwd_path: Path = Path(__file__).parent.resolve()

def summarize_data(doc_content: bytes, url: str):
    data = {"doc": doc_content}
    try:
        response: requests.Response = requests.post(f"{url}/summarize", data=data)
    except requests.exceptions.SSLError:
        raise Exception(
            f"Bad response from server: {response.status_code} {response.text}"
        )

    if response.status_code != 200:
        raise Exception(
            f"Bad response from server: {response.status_code} {response.text}"
        )

    return json.loads(response.text)


async def main(url: str, doc_path: str, self_signed_ssl: bool = False):
    response = summarize_data(open(doc_path, "rb").read(), url)

    print("Response:", response["summary"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data provider example.")
    parser.add_argument("url", type=str, help="URL of the secure API")
    parser.add_argument("doc", type=str, help="The document to summarize")

    try:
        args = parser.parse_args()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main(args.url, args.doc))
    except SystemExit:
        parser.print_help()
        raise
