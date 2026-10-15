"""LLM factory. Returns an OpenAI-compatible chat model wired to OpenRouter."""

import os

from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Neither OPENAI_API_KEY nor OPENROUTER_API_KEY is set — configure .env")

    # Determine default model and base_url
    if os.environ.get("OPENAI_API_KEY"):
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        if model.startswith("openai/"):
            model = model.removeprefix("openai/")
        base_url = os.environ.get("LLM_BASE_URL")
    else:
        model = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")
        base_url = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")

    kwargs = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
    }
    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)

