from langchain_ollama import ChatOllama

from config import settings  # Load the project's .env before creating a model.


# model_name = "qwen3.5:9b-q4_K_M"
model_name = "qwen3.5:2b-q4_K_M"


def get_model() -> ChatOllama:
    return ChatOllama(model=model_name, temperature=0, reasoning=True, num_ctx=8192)
