"""
rag_engine.py
RAG-based distress signal detection for MindBridge.

Uses TF-IDF-style cosine similarity over the mental health knowledge base
to retrieve relevant context and classify distress level from user input.
In a production IBM Cloud deployment, replace the in-memory vector store
with IBM Watson Discovery or IBM Elasticsearch on Cloud.
"""

import json
import math
import re
import os
from dataclasses import dataclass, field
from typing import Optional


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A knowledge base entry retrieved for a given query."""
    id:       str
    title:    str
    content:  str
    category: str
    score:    float


@dataclass
class DistressAnalysis:
    """Full distress analysis result for a piece of text."""
    risk_level:        str          # "none" | "low" | "medium" | "high" | "critical"
    risk_score:        float        # 0.0 – 1.0
    detected_signals:  list[str]    # specific signals found
    retrieved_context: list[RetrievedChunk]
    summary:           str          # human-readable summary
    needs_escalation:  bool         # True if human/helpline referral is needed


# ── Signal dictionaries ───────────────────────────────────────────────────────

# Critical signals → imminent risk of self-harm / suicide
CRITICAL_SIGNALS: dict[str, list[str]] = {
    "suicidal_ideation": [
        "kill myself", "end my life", "want to die", "wish i was dead",
        "rather be dead", "not want to live", "commit suicide", "take my life",
        "end it all", "no reason to live", "goodbye forever", "last message",
        "this is goodbye", "won't be here tomorrow",
    ],
    "self_harm": [
        "hurt myself", "cut myself", "harm myself", "self harm", "self-harm",
        "burning myself", "self injure",
    ],
    "means_planning": [
        "method to", "how to suicide", "pills to overdose", "jump from",
        "hanging myself", "have a plan",
    ],
}

# High signals → significant distress, imminent warning signs
HIGH_SIGNALS: dict[str, list[str]] = {
    "hopelessness": [
        "no hope", "hopeless", "nothing will change", "no future",
        "can't go on", "can't take it anymore", "pointless", "no point",
        "nothing matters", "life is meaningless",
    ],
    "burden_feeling": [
        "everyone would be better without me", "burden to others",
        "better off without me", "don't deserve to live", "nobody cares",
        "no one would miss me", "i am worthless", "i am useless",
    ],
    "isolation": [
        "completely alone", "no one understands", "no one cares about me",
        "totally isolated", "nobody there for me",
    ],
}

# Medium signals → emotional distress requiring support
MEDIUM_SIGNALS: dict[str, list[str]] = {
    "depression_symptoms": [
        "so depressed", "feel empty", "numb inside", "can't feel anything",
        "haven't gotten out of bed", "stopped eating", "lost all interest",
        "don't enjoy anything", "crying all the time", "feel dead inside",
    ],
    "anxiety_distress": [
        "panic attack", "constant anxiety", "can't breathe", "heart racing",
        "overwhelming anxiety", "paralyzed by fear", "can't function",
    ],
    "severe_stress": [
        "breaking down", "falling apart", "can't cope", "at my limit",
        "completely overwhelmed", "losing my mind", "can't handle",
    ],
}

# Low signals → mild distress / seeking support
LOW_SIGNALS: dict[str, list[str]] = {
    "general_distress": [
        "stressed", "anxious", "sad", "down", "worried", "nervous",
        "unhappy", "upset", "lonely", "tired", "exhausted", "struggling",
        "having a hard time", "not doing well", "not okay",
    ],
    "mental_health_queries": [
        "depression", "anxiety", "mental health", "therapy", "counseling",
        "coping", "feeling low", "emotional support", "need help",
    ],
}


# ── TF-IDF cosine similarity (no external dependencies) ─────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r'\b[a-z]{2,}\b', text.lower())


def _build_tfidf_vectors(
    docs: list[str],
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Build TF-IDF vectors for a list of documents."""
    tokenized = [_tokenize(d) for d in docs]

    # IDF
    N = len(docs)
    df: dict[str, int] = {}
    for tokens in tokenized:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1
    idf = {t: math.log(N / (1 + df[t])) + 1 for t in df}

    # TF-IDF
    vectors = []
    for tokens in tokenized:
        tf: dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        total = len(tokens) or 1
        vec = {t: (count / total) * idf.get(t, 1) for t, count in tf.items()}
        vectors.append(vec)

    return vectors, idf


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    dot   = sum(a.get(t, 0) * v for t, v in b.items())
    mag_a = math.sqrt(sum(v ** 2 for v in a.values())) or 1e-9
    mag_b = math.sqrt(sum(v ** 2 for v in b.values())) or 1e-9
    return dot / (mag_a * mag_b)


# ── RAG Engine ────────────────────────────────────────────────────────────────

class RAGEngine:
    """
    Retrieval-Augmented Generation engine for MindBridge.

    Responsibilities:
      1. Classify the distress level in a user message (none/low/medium/high/critical)
      2. Retrieve the most relevant knowledge-base chunks to provide as context
         to the IBM Granite model
    """

    def __init__(self, knowledge_base_path: Optional[str] = None):
        kb_path = knowledge_base_path or os.path.join(
            os.path.dirname(__file__), "..", "data", "knowledge_base.json"
        )
        with open(kb_path, "r", encoding="utf-8") as fh:
            self._kb: list[dict] = json.load(fh)

        # Pre-compute TF-IDF vectors for every knowledge base entry
        self._doc_texts  = [f"{e['title']} {e['content']} {' '.join(e['tags'])}" for e in self._kb]
        self._doc_vecs, self._idf = _build_tfidf_vectors(self._doc_texts)

    # ── Public API ───────────────────────────────────────────────────────────

    def analyze(self, text: str, conversation_history: list[dict] | None = None) -> DistressAnalysis:
        """
        Analyse text for distress signals and retrieve relevant context.

        Args:
            text:                 Latest user message.
            conversation_history: Recent conversation turns for richer context.

        Returns:
            DistressAnalysis with risk level, signals, and retrieved chunks.
        """
        # Combine recent history for richer signal detection
        combined_text = text
        if conversation_history:
            recent = " ".join(
                t.get("content", "") for t in conversation_history[-4:]
                if t.get("role") == "user"
            )
            combined_text = f"{recent} {text}"

        risk_level, risk_score, detected = self._classify_risk(combined_text)
        chunks = self._retrieve(text, top_k=3)
        summary = self._build_summary(risk_level, detected)
        needs_escalation = risk_level in ("high", "critical")

        return DistressAnalysis(
            risk_level=risk_level,
            risk_score=risk_score,
            detected_signals=detected,
            retrieved_context=chunks,
            summary=summary,
            needs_escalation=needs_escalation,
        )

    def retrieve_context_string(self, text: str, top_k: int = 3) -> str:
        """Return retrieved context as a formatted string for prompt injection."""
        chunks = self._retrieve(text, top_k)
        if not chunks:
            return ""
        parts = []
        for c in chunks:
            parts.append(f"[{c.title}]\n{c.content}")
        return "\n\n".join(parts)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _classify_risk(self, text: str) -> tuple[str, float, list[str]]:
        """Return (risk_level, risk_score, detected_signals) for the text."""
        text_lower = text.lower()
        detected: list[str] = []

        # Critical check
        for signal_name, keywords in CRITICAL_SIGNALS.items():
            for kw in keywords:
                if kw in text_lower:
                    detected.append(f"critical:{signal_name}:{kw}")
        if detected:
            return "critical", 1.0, detected

        # High check
        for signal_name, keywords in HIGH_SIGNALS.items():
            for kw in keywords:
                if kw in text_lower:
                    detected.append(f"high:{signal_name}:{kw}")
        if detected:
            return "high", 0.75 + min(0.2, len(detected) * 0.05), detected

        # Medium check
        for signal_name, keywords in MEDIUM_SIGNALS.items():
            for kw in keywords:
                if kw in text_lower:
                    detected.append(f"medium:{signal_name}:{kw}")
        if detected:
            return "medium", 0.40 + min(0.25, len(detected) * 0.05), detected

        # Low check
        for signal_name, keywords in LOW_SIGNALS.items():
            for kw in keywords:
                if kw in text_lower:
                    detected.append(f"low:{signal_name}:{kw}")
        if detected:
            return "low", 0.10 + min(0.20, len(detected) * 0.03), detected

        return "none", 0.0, []

    def _retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        """TF-IDF cosine similarity retrieval over the knowledge base."""
        q_tokens = _tokenize(query)
        total = len(q_tokens) or 1
        q_vec = {
            t: (q_tokens.count(t) / total) * self._idf.get(t, 1)
            for t in set(q_tokens)
        }

        scored: list[tuple[float, int]] = []
        for i, doc_vec in enumerate(self._doc_vecs):
            score = _cosine(q_vec, doc_vec)
            scored.append((score, i))

        scored.sort(reverse=True)
        results = []
        for score, idx in scored[:top_k]:
            if score < 0.01:
                break
            entry = self._kb[idx]
            results.append(RetrievedChunk(
                id=entry["id"],
                title=entry["title"],
                content=entry["content"],
                category=entry["category"],
                score=round(score, 4),
            ))
        return results

    def _build_summary(self, risk_level: str, signals: list[str]) -> str:
        summaries = {
            "none":     "No significant distress signals detected.",
            "low":      "Mild distress indicators present — user may benefit from supportive conversation.",
            "medium":   "Moderate distress detected — emotional support and coping resources recommended.",
            "high":     "High distress signals detected — professional referral and crisis awareness recommended.",
            "critical": "CRITICAL: Immediate crisis signals detected — helpline escalation required.",
        }
        base = summaries.get(risk_level, "Unknown risk level.")
        if signals:
            top_signals = ", ".join(s.split(":")[-1] for s in signals[:3])
            base += f" Key signals: {top_signals}."
        return base
