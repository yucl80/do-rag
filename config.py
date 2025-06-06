# Configuration file for DO-RAG project
import os
from dotenv import load_dotenv
load_dotenv() 

# LLM Provider Configurations
LLM_CONFIGS = {
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY", "your-openai-api-key"),
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1"
    },
    "ollama": {
        "model": "llama2",
        "base_url": "http://localhost:11434"
    }
}

# Default LLM provider to use
DEFAULT_LLM_PROVIDER = "openai"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
