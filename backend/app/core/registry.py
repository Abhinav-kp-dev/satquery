from __future__ import annotations

from functools import lru_cache

import yaml

from app.config import DATA_DIR
from app.models.schemas import InputConfig, Task


@lru_cache
def load_registry() -> dict:
    with open(DATA_DIR / "registry.yaml") as f:
        return yaml.safe_load(f)


def tool_plan_for(task: Task) -> list[str]:
    entry = load_registry().get(task.value, {})
    return list(entry.get("tools", []))


def is_task_applicable(task: Task, config: InputConfig) -> bool:
    entry = load_registry().get(task.value, {})
    return config.value in entry.get("applicable_configs", [])
