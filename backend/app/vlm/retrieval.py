"""Retrieval-augmented in-context learning -- the adaptation layer used in
place of gradient fine-tuning (see README: 'Why no fine-tuning'). Real
question/answer exemplars, hand-curated per task family, are retrieved by
lexical similarity to the current question and shown to the narration
model in-context, instead of the model ever updating its weights.

TF-IDF cosine similarity is used rather than a neural sentence embedder so
this stage has no model download and no GPU dependency -- consistent with
the rest of the tool layer.
"""
from __future__ import annotations

import json
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import DATA_DIR
from app.models.schemas import Task


class ExamplePool:
    def __init__(self, pool: list[dict]):
        self.pool = pool
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self.vectorizer.fit_transform([ex["question"] for ex in pool])

    def retrieve(self, question: str, task: Task, k: int = 3) -> list[dict]:
        task_idx = [i for i, ex in enumerate(self.pool) if ex["task"] == task.value]
        if not task_idx:
            return []
        q_vec = self.vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self._matrix[task_idx])[0]
        ranked = sorted(zip(task_idx, sims), key=lambda t: t[1], reverse=True)
        return [self.pool[i] for i, _ in ranked[:k]]

    def force_sufficiency_example(self) -> dict | None:
        candidates = [ex for ex in self.pool if ex["task"] == "sufficiency"]
        return candidates[0] if candidates else None


@lru_cache
def get_example_pool() -> ExamplePool:
    with open(DATA_DIR / "example_pool.json") as f:
        pool = json.load(f)
    return ExamplePool(pool)
