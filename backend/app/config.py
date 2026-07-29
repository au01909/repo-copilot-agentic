"""Central configuration, all overridable via environment variables / .env file."""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# --- LLM provider selection ---
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").lower()
# anthropic | openai | nvidia | deepseek | none

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")  # None -> official OpenAI endpoint

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
NVIDIA_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# --- Embeddings ---
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "tfidf").lower()
# tfidf | openai
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# --- Reranking ---
RERANK_PROVIDER = os.environ.get("RERANK_PROVIDER", "lexical").lower()
# lexical | cohere | none
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
COHERE_RERANK_MODEL = os.environ.get("COHERE_RERANK_MODEL", "rerank-english-v3.0")

# --- Retrieval tuning ---
FETCH_K = int(os.environ.get("FETCH_K", "50"))
TOP_K = int(os.environ.get("TOP_K", "3"))
BM25_WEIGHT = _float("BM25_WEIGHT", 1.0)
DENSE_WEIGHT = _float("DENSE_WEIGHT", 1.0)
MIN_BM25_SCORE = _float("MIN_BM25_SCORE", 0.0)
MIN_DENSE_SCORE = _float("MIN_DENSE_SCORE", 0.05)
DEDUPE_ONE_CHUNK_PER_FILE = _bool("DEDUPE_ONE_CHUNK_PER_FILE", True)

# --- Vector store ---
VECTOR_STORE = os.environ.get("VECTOR_STORE", "qdrant_memory").lower()
# qdrant_memory | qdrant_local (on-disk) | qdrant_server (remote URL)
QDRANT_PATH = os.environ.get("QDRANT_PATH", "./qdrant_data")
QDRANT_URL = os.environ.get("QDRANT_URL")  # for qdrant_server mode
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

# --- Persistence ---
SESSION_STORE = os.environ.get("SESSION_STORE", "sqlite").lower()
# sqlite | memory  (adapter pattern; swap in Redis/Postgres by implementing SessionStore)
SQLITE_PATH = os.environ.get("SQLITE_PATH", "./repo_copilot.db")

# --- GitHub integration ---
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")  # optional, raises rate limits

# --- Feature flags ---
ENABLE_CODE_EXECUTION = _bool("ENABLE_CODE_EXECUTION", False)  # off by default: arbitrary repo code execution is inherently risky
ENABLE_MULTI_QUERY = _bool("ENABLE_MULTI_QUERY", True)
ENABLE_QUERY_EXPANSION = _bool("ENABLE_QUERY_EXPANSION", True)
ENABLE_AGENTIC_PLANNER = _bool("ENABLE_AGENTIC_PLANNER", True)

# ---------------------------------------------------------------------------
# Agentic-platform upgrade: LLM reliability, ADK/LangGraph/ChatNVIDIA, MLflow,
# LangSmith, ingestion limits. Everything below is additive — nothing above
# this line changes meaning or default behavior.
# ---------------------------------------------------------------------------

# --- NVIDIA AI Endpoints (ChatNVIDIA, via app/llm/nvidia_provider.py) ---
# Distinct from NVIDIA_MODEL/NVIDIA_BASE_URL above, which are for the existing
# OpenAI-compatible provider path in app/providers.py. This is the LangChain-
# native integration path used by the ADK agent / LangGraph workflow.
LLM_MODEL = os.environ.get("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.2)
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "4096"))
ENABLE_REASONING = _bool("ENABLE_REASONING", False)
REASONING_BUDGET = int(os.environ.get("REASONING_BUDGET", "4096"))

# --- LLM reliability (retry / backoff / fallback) ---
LLM_TIMEOUT = _float("LLM_TIMEOUT", 30.0)
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "2"))
LLM_FALLBACK_PROVIDER = os.environ.get("LLM_FALLBACK_PROVIDER", "").lower() or None

# --- Repository agent framework selection ---
AGENT_FRAMEWORK = os.environ.get("AGENT_FRAMEWORK", "adk").lower()
# adk (Google ADK Repository Agent, default) | direct (call the LangGraph
# workflow directly, skipping the ADK tool-calling wrapper — useful if ADK or
# its model backend isn't configured, e.g. no NVIDIA_API_KEY)
AGENT_MAX_RETRIES = int(os.environ.get("AGENT_MAX_RETRIES", "2"))

# --- Repository ingestion limits (untrusted input) ---
MAX_REPOSITORY_SIZE_MB = int(os.environ.get("MAX_REPOSITORY_SIZE", "500"))
MAX_FILES = int(os.environ.get("MAX_FILES", "5000"))
MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE", "500000"))
MAX_CHUNKS = int(os.environ.get("MAX_CHUNKS", "20000"))
CLONE_TIMEOUT = int(os.environ.get("CLONE_TIMEOUT", "180"))
INDEX_TIMEOUT = int(os.environ.get("INDEX_TIMEOUT", "300"))

# --- Memory / persistence ---
REDIS_URL = os.environ.get("REDIS_URL")  # if set, memory/store.py prefers Redis; else SQLite/in-memory
MAX_HISTORY_TURNS_SENT_TO_LLM = int(os.environ.get("MAX_HISTORY_TURNS_SENT_TO_LLM", "6"))

# --- CORS / internal endpoints ---
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
INTERNAL_API_TOKEN = os.environ.get("INTERNAL_API_TOKEN")

# --- MLflow (local experiment tracking; safe no-op if unset/unreachable) ---
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///./mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "repo-copilot")
ENABLE_MLFLOW = _bool("ENABLE_MLFLOW", True)

# --- LangSmith (purely env-var driven by LangChain itself; nothing to call) ---
# Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY=... to enable. The app
# never checks these directly — LangChain/LangGraph read them automatically.

APP_ENV = os.environ.get("APP_ENV", "development")
