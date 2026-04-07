"""Configuration and API key loading."""

import os

from dotenv import load_dotenv

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACE_API_KEY", "")

# LLM provider API keys (via litellm)
API_MODELS = {
    "deepseek/deepseek-chat": {
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    },
    "gpt-4.1-nano": {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
    },
    "gpt-5-nano": {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
    },
    "gemini/gemini-2.5-flash-lite": {
        "api_key": os.getenv("GEMINI_API_KEY", ""),
    },
    "mistral/mistral-small": {
        "api_key": os.getenv("MISTRAL_API_KEY", ""),
    },
    "mistral/mistral-tiny": {
        "api_key": os.getenv("MISTRAL_API_KEY", ""),
    },
    "mistral/pixtral-12b-2409": {
        "api_key": os.getenv("MISTRAL_API_KEY", ""),
    },
    "groq/llama-3.1-8b-instant": {
        "api_key": os.getenv("GROQ_API_KEY", ""),
    },
    "groq/meta-llama/llama-4-scout-17b-16e-instruct": {
        "api_key": os.getenv("GROQ_API_KEY", ""),
    },
    "groq/openai/gpt-oss-120b": {
        "api_key": os.getenv("GROQ_API_KEY", ""),
    },
}


BLACKLIST_FIELDS = {
    "id",
    "uuid",
    "_id",
    "object_id",
    "timestamp",
    "created_at",
    "updated_at",
}
