import os

import requests

from env_config import load_env_file


GEMINI_DEFAULT_MODEL = "gemma-4-31b-it"
LOCAL_DEFAULT_MODEL = "llama3.2"
LOCAL_BASE_URL = "http://localhost:11434"

PROVIDERS = ("gemini", "local")


class LLMError(RuntimeError):
    pass


def query_llm(
    prompt: str,
    provider: str = "gemini",
    model: str | None = None,
    system_prompt: str | None = None,
) -> str:
    if provider == "gemini":
        return _query_gemini(prompt, model=model, system_prompt=system_prompt)
    if provider == "local":
        return _query_local(prompt, model=model, system_prompt=system_prompt)
    raise ValueError(f"Unsupported LLM provider: {provider}")


def _query_gemini(prompt: str, model: str | None, system_prompt: str | None) -> str:
    load_env_file()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY is required for Gemini provider. Add it to .env")

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise LLMError(
            "google-genai is required for Gemini provider. "
            "Install with: pip install google-genai"
        )

    client = genai.Client(api_key=api_key)
    model_name = model or GEMINI_DEFAULT_MODEL

    config = None
    if system_prompt:
        config = types.GenerateContentConfig(system_instruction=system_prompt)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config,
    )
    return response.text


def _query_local(prompt: str, model: str | None, system_prompt: str | None) -> str:
    model_name = model or LOCAL_DEFAULT_MODEL
    url = f"{LOCAL_BASE_URL}/api/chat"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]
    except requests.RequestException as exc:
        raise LLMError(f"Local LLM request failed: {exc}")
