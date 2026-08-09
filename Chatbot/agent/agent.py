from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from agent.ollama_cloud import build_ollama_llm
from agent.tools import calculator, get_current_datetime

SYSTEM_PROMPT = """You are a helpful AI assistant.
Answer clearly and concisely. Use tools when needed for calculations or the current time.
If the user writes in Azerbaijani or Turkish, respond in the same language."""


def build_agent(provider: str, model_name: str):
    if provider == "OpenAI":
        llm = ChatOpenAI(model=model_name, temperature=0.7)
    else:
        llm = build_ollama_llm(model_name)

    return create_agent(
        llm,
        tools=[get_current_datetime, calculator],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
    )
