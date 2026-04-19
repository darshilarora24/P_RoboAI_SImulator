"""
rag_engine.py  —  Robot Knowledge Base with RAG using Gemini Embeddings.

Uses google-generativeai text-embedding-004 model to embed robot knowledge
(URDF specs, task docs, sensor history, trajectory logs) and retrieves the
most relevant context chunks for grounding Gemini robot task responses.

Falls back to TF-IDF cosine similarity if no API key is set.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── optional deps ─────────────────────────────────────────────────────────────

try:
    import google.generativeai as genai
    _GENAI_OK = True
except ImportError:
    _GENAI_OK = False

try:
    import numpy as np
    _NP_OK = True
except ImportError:
    _NP_OK = False


# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class KnowledgeChunk:
    id: str
    text: str
    source: str          # "urdf", "task_doc", "sensor_log", "trajectory"
    metadata: dict = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)


@dataclass
class RetrievalResult:
    chunk: KnowledgeChunk
    score: float


# ── TF-IDF fallback ───────────────────────────────────────────────────────────

class _TFIDF:
    """Minimal TF-IDF for when Gemini API is unavailable."""

    def __init__(self) -> None:
        self._docs: list[list[str]] = []
        self._idf: dict[str, float] = {}

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def fit(self, texts: list[str]) -> None:
        self._docs = [self._tokenize(t) for t in texts]
        N = len(self._docs)
        df: dict[str, int] = {}
        for doc in self._docs:
            for w in set(doc):
                df[w] = df.get(w, 0) + 1
        self._idf = {w: math.log((N + 1) / (d + 1)) + 1.0 for w, d in df.items()}

    def score(self, query: str, doc_idx: int) -> float:
        q_toks = self._tokenize(query)
        doc = self._docs[doc_idx]
        if not doc or not q_toks:
            return 0.0
        tf: dict[str, float] = {}
        for w in doc:
            tf[w] = tf.get(w, 0) + 1
        total = sum(
            (tf.get(w, 0) / len(doc)) * self._idf.get(w, 0.0)
            for w in q_toks
        )
        return total


# ── main class ────────────────────────────────────────────────────────────────

class RobotKnowledgeBase:
    """
    Vector knowledge base for robot task grounding.

    Usage
    -----
      kb = RobotKnowledgeBase(api_key="AIza...")
      kb.add_urdf_spec("arm", urdf_xml_string)
      kb.add_task_doc("pick_place", "The robot picks object A and places...")
      results = kb.retrieve("how do I pick a box?", top_k=3)
    """

    EMBED_MODEL = "models/text-embedding-004"
    EMBED_DIM   = 768

    def __init__(self, api_key: str = "", persist_path: str = "") -> None:
        self._chunks: list[KnowledgeChunk] = []
        self._tfidf  = _TFIDF()
        self._dirty  = True
        self._persist = Path(persist_path) if persist_path else None

        self._use_gemini = False
        if _GENAI_OK and (api_key or os.environ.get("GOOGLE_API_KEY")):
            key = api_key or os.environ["GOOGLE_API_KEY"]
            genai.configure(api_key=key)
            self._use_gemini = True

        if self._persist and self._persist.exists():
            self._load_persist()

    # ── ingestion ─────────────────────────────────────────────────────────────

    def add_urdf_spec(self, robot_name: str, urdf_xml: str) -> None:
        """Chunk URDF XML into link/joint sections and embed each."""
        chunks = self._chunk_urdf(robot_name, urdf_xml)
        for c in chunks:
            self._add_chunk(c)

    def add_task_doc(self, task_name: str, doc_text: str) -> None:
        """Add a freeform task description document."""
        for i, para in enumerate(self._split_paragraphs(doc_text)):
            cid = self._make_id(f"task:{task_name}:{i}")
            self._add_chunk(KnowledgeChunk(
                id=cid, text=para, source="task_doc",
                metadata={"task": task_name, "para": i}
            ))

    def add_sensor_snapshot(self, sensor_data: dict) -> None:
        """Add a sensor state snapshot (joint positions, odom, etc.)."""
        text = "Robot sensor snapshot: " + json.dumps(sensor_data, indent=2)
        cid  = self._make_id(text)
        self._add_chunk(KnowledgeChunk(
            id=cid, text=text[:1000], source="sensor_log",
            metadata=sensor_data
        ))

    def add_trajectory(self, traj_name: str, waypoints: list[dict]) -> None:
        """Add a named trajectory as a knowledge chunk."""
        text = (f"Trajectory '{traj_name}' with {len(waypoints)} waypoints: "
                + json.dumps(waypoints[:5]))
        cid  = self._make_id(f"traj:{traj_name}")
        self._add_chunk(KnowledgeChunk(
            id=cid, text=text, source="trajectory",
            metadata={"name": traj_name, "length": len(waypoints)}
        ))

    # ── retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not self._chunks:
            return []

        if self._use_gemini and _NP_OK:
            return self._retrieve_gemini(query, top_k)
        return self._retrieve_tfidf(query, top_k)

    def format_context(self, results: list[RetrievalResult]) -> str:
        lines = ["=== Retrieved Robot Knowledge ==="]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] Source: {r.chunk.source} | Score: {r.score:.3f}")
            lines.append(r.chunk.text[:500])
        return "\n".join(lines)

    # ── internals ─────────────────────────────────────────────────────────────

    def _add_chunk(self, chunk: KnowledgeChunk) -> None:
        # Deduplicate by id
        existing = {c.id for c in self._chunks}
        if chunk.id in existing:
            return
        if self._use_gemini and _NP_OK and not chunk.embedding:
            try:
                res = genai.embed_content(
                    model=self.EMBED_MODEL,
                    content=chunk.text,
                    task_type="retrieval_document",
                )
                chunk.embedding = res["embedding"]
            except Exception:
                pass
        self._chunks.append(chunk)
        self._dirty = True

    def _retrieve_gemini(self, query: str, top_k: int) -> list[RetrievalResult]:
        try:
            res = genai.embed_content(
                model=self.EMBED_MODEL,
                content=query,
                task_type="retrieval_query",
            )
            q_emb = np.array(res["embedding"], dtype=np.float32)
        except Exception:
            return self._retrieve_tfidf(query, top_k)

        scored = []
        for chunk in self._chunks:
            if not chunk.embedding:
                continue
            c_emb = np.array(chunk.embedding, dtype=np.float32)
            norm  = np.linalg.norm(q_emb) * np.linalg.norm(c_emb)
            score = float(np.dot(q_emb, c_emb) / norm) if norm > 0 else 0.0
            scored.append(RetrievalResult(chunk=chunk, score=score))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def _retrieve_tfidf(self, query: str, top_k: int) -> list[RetrievalResult]:
        if self._dirty:
            self._tfidf.fit([c.text for c in self._chunks])
            self._dirty = False
        scored = [
            RetrievalResult(chunk=self._chunks[i],
                            score=self._tfidf.score(query, i))
            for i in range(len(self._chunks))
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def _chunk_urdf(self, robot_name: str, urdf_xml: str) -> list[KnowledgeChunk]:
        chunks = []
        # Split by joint/link tags
        for tag in ("link", "joint"):
            for m in re.finditer(rf"<{tag}[^>]*>.*?</{tag}>", urdf_xml,
                                 re.DOTALL):
                snippet = m.group(0)[:600]
                name_m  = re.search(r'name="([^"]+)"', snippet)
                name    = name_m.group(1) if name_m else "unknown"
                cid     = self._make_id(f"urdf:{robot_name}:{tag}:{name}")
                chunks.append(KnowledgeChunk(
                    id=cid,
                    text=f"Robot {robot_name} {tag} '{name}':\n{snippet}",
                    source="urdf",
                    metadata={"robot": robot_name, "element": tag, "name": name},
                ))
        if not chunks:
            cid = self._make_id(f"urdf:{robot_name}:full")
            chunks.append(KnowledgeChunk(
                id=cid, text=urdf_xml[:1000], source="urdf",
                metadata={"robot": robot_name}
            ))
        return chunks

    @staticmethod
    def _split_paragraphs(text: str, max_len: int = 600) -> list[str]:
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        result = []
        for p in paras:
            while len(p) > max_len:
                result.append(p[:max_len])
                p = p[max_len:]
            if p:
                result.append(p)
        return result or [text[:max_len]]

    @staticmethod
    def _make_id(seed: str) -> str:
        return hashlib.sha1(seed.encode()).hexdigest()[:16]

    def _load_persist(self) -> None:
        try:
            data = json.loads(self._persist.read_text())
            for d in data:
                self._chunks.append(KnowledgeChunk(**d))
            self._dirty = True
        except Exception:
            pass

    def save(self) -> None:
        if self._persist:
            data = [
                {"id": c.id, "text": c.text, "source": c.source,
                 "metadata": c.metadata, "embedding": c.embedding}
                for c in self._chunks
            ]
            self._persist.write_text(json.dumps(data))
