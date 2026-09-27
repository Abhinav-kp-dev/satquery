from __future__ import annotations

from functools import lru_cache

import yaml

from app.config import DATA_DIR
from app.models.schemas import InputConfig, Task


@lru_cache
def load_registry() -> dict:
    with open(DATA_DIR / "registry.yaml") as f:
        return yaml.safe_load(f)


def task_entry(task: Task) -> dict:
    return load_registry()["tasks"].get(task.value, {})


def tool_entry(tool: str) -> dict:
    return load_registry()["tools"].get(tool, {})


def allowed_tools(task: Task) -> list[str]:
    return list(task_entry(task).get("tools", []))


def adapter_for(task: Task) -> str | None:
    return task_entry(task).get("adapter")


def is_task_applicable(task: Task, config: InputConfig) -> bool:
    return config.value in task_entry(task).get("applicable_configs", [])
