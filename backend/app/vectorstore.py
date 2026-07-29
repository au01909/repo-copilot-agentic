"""Vector store backed by Qdrant.

Qdrant supports an embedded mode with no external server (`:memory:` or a local
on-disk path), which is what makes this genuinely runnable in a sandbox without
provisioning infrastructure. Set VECTOR_STORE=qdrant_server + QDRANT_URL to point
at a real Qdrant cluster in production — same interface, no code changes needed
elsewhere.
"""
import uuid
from typing import Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from . import config


def _make_client() -> QdrantClient:
    mode = config.VECTOR_STORE
    if mode == "qdrant_server" and config.QDRANT_URL:
        return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    if mode == "qdrant_local":
        return QdrantClient(path=config.QDRANT_PATH)
    return QdrantClient(":memory:")  # qdrant_memory (default) — fresh per process


class VectorStore:
    def __init__(self, collection_name: str, dim: int):
        self.client = _make_client()
        self.collection_name = collection_name
        self.dim = dim
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )
        self._id_map: Dict[str, int] = {}   # external chunk id -> vector index

    def upsert(self, ids: List[str], vectors, payloads: List[dict]):
        points = []
        for i, (ext_id, vec, payload) in enumerate(zip(ids, vectors, payloads)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, ext_id))
            self._id_map[point_id] = i
            payload = dict(payload)
            payload["_ext_id"] = ext_id
            points.append(qmodels.PointStruct(id=point_id, vector=vec.tolist(), payload=payload))
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector, top_k: int) -> List[Tuple[str, float, dict]]:
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
        )
        hits = result.points
        return [(h.payload.get("_ext_id"), h.score, h.payload) for h in hits]

    def delete(self, ids: List[str]):
        point_ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, i)) for i in ids]
        if point_ids:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.PointIdsList(points=point_ids),
            )

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass
