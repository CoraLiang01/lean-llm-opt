#!/usr/bin/env python3
"""Check model server without installing dependencies or running inference."""
from __future__ import annotations

import argparse
import os
import sys

from run_airnrm_np import MissingModelsError, api, normalize_ollama_url, preflight


def diagnose(value: str) -> int:
    value = value.strip().rstrip("/")
    ollama_error = None
    if not value.endswith("/v1"):
        try:
            base = normalize_ollama_url(value)
            preflight(base)
            print("PASS: Ollama service and both required model tags found. No inference was called.")
            return 0
        except MissingModelsError as exc:
            print("API TYPE: Ollama. " + str(exc))
            return 1
        except (RuntimeError, ValueError) as exc:
            ollama_error = exc
    # Identify a different API instead of calling every endpoint failure a
    # missing model. An OpenAI-compatible API does not identify its backend.
    openai_base = value[:-4] if value.endswith("/api") else value
    endpoint = "/models" if openai_base.endswith("/v1") else "/v1/models"
    try:
        response = api(openai_base, endpoint)
        models = response.get("data") if isinstance(response, dict) else None
        if isinstance(models, list):
            names = [model.get("id") for model in models if isinstance(model, dict)]
            print("API TYPE: OpenAI-compatible (possibly vLLM/SGLang or another service).")
            print("Advertised models: " + ", ".join(str(name) for name in names))
            print("This NP package currently requires Ollama for both chat and embeddings. "
                  "Use an Ollama endpoint, or have the package adapted for this API first.")
            return 2
    except (RuntimeError, ValueError):
        pass
    print("CHECK FAILED: " + str(ollama_error or
          "The /v1 endpoint could not be verified. Check the address, access and API authentication."))
    print("No model was downloaded or invoked. Set --url to the actual inference service address.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=(os.environ.get("OLLAMA_BASE_URL")
                                         or os.environ.get("OLLAMA_URL")
                                         or "http://localhost:11434"))
    args = parser.parse_args()
    print("Checking server: " + args.url)
    return diagnose(args.url)


if __name__ == "__main__":
    sys.exit(main())
