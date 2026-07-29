"""Heuristic architecture detection.

No ML model needed: infers likely architecture patterns from folder naming
conventions, framework imports, and structural signals. This is intentionally
a heuristic (fast, explainable, zero external dependencies) rather than an
LLM classification — good enough to seed the "Explain the architecture" answer,
and cheap to run on every index.
"""
import re
from collections import Counter
from typing import Dict, List

from .chunking import Chunk

FOLDER_SIGNALS = {
    "layered/mvc": ["controllers", "models", "views", "services", "repositories"],
    "microservices": ["services", "gateway", "docker-compose"],
    "hexagonal": ["adapters", "ports", "domain"],
    "event-driven": ["consumers", "producers", "events", "handlers"],
}

FRAMEWORK_SIGNALS = {
    "FastAPI": [r"from fastapi", r"import fastapi"],
    "Flask": [r"from flask", r"import flask"],
    "Django": [r"from django", r"import django"],
    "Express": [r"require\(['\"]express['\"]\)", r"from ['\"]express['\"]"],
    "Spring": [r"@SpringBootApplication", r"org\.springframework"],
    "Kafka": [r"kafka", r"KafkaProducer", r"KafkaConsumer"],
    "gRPC": [r"grpc", r"\.proto\b"],
    "REST": [r"@app\.route", r"@app\.get", r"@app\.post", r"@RestController"],
    "GraphQL": [r"graphql", r"gql`"],
    "Redis": [r"redis"],
    "SQL/ORM": [r"sqlalchemy", r"SELECT .* FROM", r"@Entity", r"models\.Model"],
}


def detect(chunks: List[Chunk]) -> Dict:
    folder_hits = Counter()
    for c in chunks:
        parts = c.file.lower().split("/")
        for pattern, keywords in FOLDER_SIGNALS.items():
            if any(kw in parts for kw in keywords):
                folder_hits[pattern] += 1

    framework_hits = Counter()
    sample_text = "\n".join(c.content for c in chunks)[:500_000]  # cap for perf
    for framework, patterns in FRAMEWORK_SIGNALS.items():
        for pat in patterns:
            if re.search(pat, sample_text, re.IGNORECASE):
                framework_hits[framework] += 1
                break

    likely_patterns = [p for p, _ in folder_hits.most_common(3)]
    frameworks = [f for f, _ in framework_hits.most_common()]

    return {
        "likely_architecture_patterns": likely_patterns or ["unclear — no strong folder-naming signal"],
        "detected_frameworks_and_infra": frameworks or ["none detected"],
        "signal_strength": dict(folder_hits),
    }
