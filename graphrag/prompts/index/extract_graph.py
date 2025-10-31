# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Prompt definitions for graph extraction."""

import os
from pathlib import Path
from typing import Dict, List, Optional

_PROMPT_CFG_ENV = "GRAPHRAG_EXTRACT_GRAPH_PROMPT_CFG"
_PROMPT_CFG_DIR = Path(__file__).resolve().parents[3] / "prompts_cfg"
_DEFAULT_CFG_NAME = "extract_graph_default"


def _resolve_config_path(config_name: str) -> Path:
    """Resolve a prompt configuration name or path to an absolute YAML path."""
    candidate = Path(config_name)
    if not candidate.suffix:
        candidate = candidate.with_suffix(".yml")
    if not candidate.is_absolute():
        candidate = _PROMPT_CFG_DIR / candidate
    return candidate


def _finalize_block(lines: List[str]) -> str:
    """Join collected block lines respecting YAML's strip behavior for |-."""
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _load_yaml_prompts(path: Path) -> Dict[str, str]:
    """Load prompt strings from a minimal YAML subset (mapping of block scalars)."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    prompts: Dict[str, str] = {}
    key: Optional[str] = None
    block_lines: List[str] = []
    block_indent: Optional[int] = None
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if key is None:
            if not stripped or stripped.startswith("#"):
                index += 1
                continue

            if ":" not in line:
                raise ValueError(f"Invalid prompt configuration line in {path}: '{line}'")

            key_part, value_part = line.split(":", 1)
            candidate_key = key_part.strip()
            value = value_part.strip()

            if value in {"|-", "|"}:
                key = candidate_key
                block_lines = []
                block_indent = None
                index += 1
                continue

            prompts[candidate_key] = value
            index += 1
            continue

        # Collecting block content
        if block_indent is None:
            if stripped == "":
                block_lines.append("")
                index += 1
                continue
            block_indent = len(line) - len(line.lstrip(" "))

        current_indent = len(line) - len(line.lstrip(" "))
        if stripped and current_indent < (block_indent or 0):
            prompts[key] = _finalize_block(block_lines)
            key = None
            block_lines = []
            block_indent = None
            # reprocess this line without advancing the index
            continue

        if block_indent:
            content = line[block_indent:]
        else:
            content = line.lstrip(" ")
        block_lines.append(content)
        index += 1

    if key is not None:
        prompts[key] = _finalize_block(block_lines)

    return prompts


def _load_prompts() -> Dict[str, str]:
    """Load prompts from YAML, applying environment overrides if provided."""
    default_path = _resolve_config_path(_DEFAULT_CFG_NAME)
    if not default_path.exists():
        raise FileNotFoundError(
            f"Default prompt configuration '{default_path}' was not found."
        )

    prompts = _load_yaml_prompts(default_path)

    config_name = os.getenv(_PROMPT_CFG_ENV)
    if not config_name:
        required = ("graph_extraction_prompt", "continue_prompt", "loop_prompt")
        missing = [key for key in required if key not in prompts]
        if missing:
            raise ValueError(
                "Default prompt configuration is missing required keys: "
                + ", ".join(missing)
            )
        return prompts

    override_path = _resolve_config_path(config_name)
    if not override_path.exists():
        raise FileNotFoundError(
            "Prompt configuration file "
            f"'{override_path}' was not found. Set {_PROMPT_CFG_ENV} to a valid YAML file."
        )

    prompts.update(_load_yaml_prompts(override_path))
    required = ("graph_extraction_prompt", "continue_prompt", "loop_prompt")
    missing = [key for key in required if key not in prompts]
    if missing:
        raise ValueError(
            "Prompt configuration is missing required keys after applying overrides: "
            + ", ".join(missing)
        )
    return prompts


_PROMPTS = _load_prompts()

GRAPH_EXTRACTION_PROMPT = _PROMPTS["graph_extraction_prompt"]
CONTINUE_PROMPT = _PROMPTS["continue_prompt"]
LOOP_PROMPT = _PROMPTS["loop_prompt"]

__all__ = [
    "GRAPH_EXTRACTION_PROMPT",
    "CONTINUE_PROMPT",
    "LOOP_PROMPT",
]
