"""MLflow tracking for repository-query runs.

Uses a local SQLite-backed tracking store by default (`MLFLOW_TRACKING_URI`,
default `sqlite:///./mlflow.db`) — no MLflow server needed to get real
experiment tracking; `mlflow ui --backend-store-uri sqlite:///./mlflow.db`
opens it. Every function here fails soft: if MLflow isn't installed, the
tracking URI isn't writable, or logging throws for any reason, the query
still completes normally — tracking must never break the app it's tracking.
"""
import logging
import time
from contextlib import contextmanager
from typing import Optional

from .. import config

logger = logging.getLogger("repo_copilot.mlflow")

_mlflow = None
_initialized = False


def _get_mlflow():
    global _mlflow, _initialized
    if _initialized:
        return _mlflow
    _initialized = True
    if not config.ENABLE_MLFLOW:
        return None
    try:
        import mlflow
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
        _mlflow = mlflow
    except Exception as e:
        logger.warning(f"MLflow unavailable, tracking disabled for this process: {e}")
        _mlflow = None
    return _mlflow


@contextmanager
def track_query(session_id: str, question: str, provider: str, model: str, intent: Optional[str] = None):
    """Wraps one /api/chat call. Usage:

        with track_query(session_id, question, provider, model) as run:
            ... do the work ...
            run.log(retrieved_chunks=5, retrieval_latency=0.01, llm_latency=0.8, mode="generative")

    `run.log(...)` is itself best-effort — never raises even if MLflow is down.
    """
    mlflow = _get_mlflow()
    start = time.time()

    class _RunLogger:
        def log(self, **metrics):
            if mlflow is None:
                return
            try:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                    else:
                        mlflow.log_param(key, str(value)[:250])
            except Exception as e:
                logger.warning(f"MLflow logging failed (non-fatal): {e}")

    run_logger = _RunLogger()

    if mlflow is None:
        yield run_logger
        return

    try:
        with mlflow.start_run(run_name=f"query-{session_id[:8]}"):
            mlflow.log_param("session_id", session_id)
            mlflow.log_param("question", question[:250])
            mlflow.log_param("provider", provider)
            mlflow.log_param("model", model)
            if intent:
                mlflow.log_param("intent", intent)
            yield run_logger
            mlflow.log_metric("total_latency_seconds", round(time.time() - start, 3))
    except Exception as e:
        logger.warning(f"MLflow run failed (non-fatal, query still completes): {e}")
        yield run_logger
