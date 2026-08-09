import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from ollama import Client

load_dotenv()

OLLAMA_HOST = "https://ollama.com"
DEFAULT_OLLAMA_MODEL = "gpt-oss:120b-cloud"


def get_ollama_client() -> Client:
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        raise ValueError("OLLAMA_API_KEY tapilmadi. .env faylini yoxlayin.")

    return Client(
        host=OLLAMA_HOST,
        headers={"Authorization": f"Bearer {api_key}"},
    )


def build_ollama_llm(model_name: str) -> ChatOllama:
    """LangChain agent ucun eyni bulud konfiqurasiyasi."""
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        raise ValueError("OLLAMA_API_KEY tapilmadi. .env faylini yoxlayin.")

    get_ollama_client()

    return ChatOllama(
        model=model_name,
        base_url=OLLAMA_HOST,
        client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}},
        temperature=0.7,
    )


def chat(messages: list[dict], model: str = DEFAULT_OLLAMA_MODEL) -> str:
    client = get_ollama_client()
    response = client.chat(model=model, messages=messages)
    return response["message"]["content"]
