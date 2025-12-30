#!/usr/bin/env python
import os
import sys
import json
import argparse
import logging
import textwrap
from config import OPENROUTER_API_KEY
from typing import Any

import requests


OPENROUTER_CHAT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def extract_content_from_choice(choice: dict) -> tuple[str, str]:
    """
    Try to extract (content, reasoning) from a single choice in the OpenRouter
    chat-style response, being robust to:
      - message.content as string
      - message.content as list of blocks, including 'reasoning' blocks
    """
    content = ""
    reasoning = ""

    msg = choice.get("message")
    if isinstance(msg, dict) and "content" in msg:
        raw_content = msg["content"]

        # Case 1: simple string
        if isinstance(raw_content, str):
            content = raw_content

        # Case 2: list of blocks (new 'reasoning' format)
        elif isinstance(raw_content, list):
            text_chunks = []
            reasoning_chunks = []
            for part in raw_content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                text = part.get("text", "")

                if ptype in ("reasoning", "chain_of_thought"):
                    reasoning_chunks.append(text)
                else:
                    text_chunks.append(text)

            content = "\n".join(c for c in text_chunks if c).strip()
            if reasoning_chunks:
                reasoning = "\n\n".join(r for r in reasoning_chunks if r).strip()

    # Fallback: completion-style 'text'
    if not content:
        raw_text = choice.get("text")
        if isinstance(raw_text, str):
            content = raw_text

    if not content:
        content = str(choice)

    return content, reasoning


def main():
    parser = argparse.ArgumentParser(
        description="Minimal OpenRouter chat test script (raw JSON inspector)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/o4-mini-high-2025-04-16",
        help="Model ID to query (OpenRouter format).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write a tiny Python function that prints 'hello from OpenRouter!'.",
        help="User prompt to send.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens in completion.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    api_key = config.OPENROUTER_API_KEY
    if not api_key:
        logging.error("OPENROUTER_API_KEY is not set in the environment.")
        sys.exit(1)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": args.model,
        "messages": [
            {"role": "user", "content": args.prompt}
        ],
        "usage": {"include": True},
        "max_tokens": args.max_tokens,
        # Many models ignore this, but reasoning models may honour it.
        "reasoning": {"exclude": False, "effort": "high"},
    }

    logging.info("Sending request to %s", OPENROUTER_CHAT_ENDPOINT)
    logging.info("Model: %s", args.model)

    try:
        resp = requests.post(
            OPENROUTER_CHAT_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=240,
        )
    except Exception as e:
        logging.exception("Request failed: %s", e)
        sys.exit(1)

    logging.info("HTTP status: %s", resp.status_code)
    raw_text = resp.text
    logging.debug("Raw body (first 1k chars):\n%s", raw_text[:1000])

    try:
        data: dict[str, Any] = resp.json()
    except json.JSONDecodeError as e:
        logging.error("Failed to decode JSON: %s", e)
        print("Raw response text:\n", raw_text)
        sys.exit(1)

    print("\n=== Pretty JSON (truncated to 4000 chars) ===")
    pretty = json.dumps(data, indent=2, ensure_ascii=False)
    print(pretty[:4000])
    if len(pretty) > 4000:
        print("... [truncated]")

    # Inspect choices structure
    choices = data.get("choices")
    print("\n=== choices type ===")
    print(type(choices), f"(len={len(choices) if isinstance(choices, list) else 'N/A'})")

    if not choices:
        print("\nNo choices returned in JSON.")
        sys.exit(0)

    first_choice = choices[0]
    print("\n=== First choice keys ===")
    print(first_choice.keys())

    msg = first_choice.get("message")
    print("\n=== message field ===")
    print("type:", type(msg))
    if isinstance(msg, dict):
        print("message keys:", msg.keys())
        print("message.content type:", type(msg.get("content")))
        if isinstance(msg.get("content"), list):
            print("message.content[0:3] sample:", msg["content"][:3])

    # Try to extract content + reasoning with robust helper
    content, reasoning = extract_content_from_choice(first_choice)

    print("\n=== Extracted content (first 1500 chars) ===")
    print(textwrap.shorten(content, width=1500, placeholder=" ... [truncated]"))

    print("\n=== Extracted reasoning (first 1500 chars) ===")
    if reasoning:
        print(textwrap.shorten(reasoning, width=1500, placeholder=" ... [truncated]"))
    else:
        print("[no separate reasoning extracted]")


if __name__ == "__main__":
    main()
