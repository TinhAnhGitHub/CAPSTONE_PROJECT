"""Dataset loaders for VideoDeepSearch evaluation.

Provides utilities for loading and saving evaluation datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from videodeepsearch.evaluation.datasets.builder import DatasetBuilder
from videodeepsearch.evaluation.datasets.schema import EvalRecord


def load_from_json(path: Path | str) -> DatasetBuilder:
    """Load dataset from JSON file.

    Args:
        path: Path to JSON file

    Returns:
        DatasetBuilder with loaded records
    """
    builder = DatasetBuilder(name="loaded_dataset")
    builder.load_from_json(path)
    return builder


def load_from_yaml(path: Path | str) -> DatasetBuilder:
    """Load dataset from YAML file.

    Expected YAML format:
    ```yaml
    name: my_dataset
    description: Description
    version: "1.0"
    records:
      - inputs:
          user_id: user1
          video_ids: [video1, video2]
          user_demand: "Find information about X"
        expectations:
          expected_response: "The answer is..."
      - inputs:
          ...
    ```

    Args:
        path: Path to YAML file

    Returns:
        DatasetBuilder with loaded records
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    name = data.get("name", "loaded_dataset")
    description = data.get("description")
    version = data.get("version")

    builder = DatasetBuilder(
        name=name,
        description=description,
        version=version,
    )

    records_data = data.get("records", [])
    if isinstance(records_data, list):
        builder.add_records_from_list(records_data)

    return builder


def save_to_json(
    builder: DatasetBuilder,
    path: Path | str,
    indent: int = 2,
) -> None:
    """Save dataset to JSON file.

    Args:
        builder: DatasetBuilder to save
        path: Output file path
        indent: JSON indentation
    """
    builder.save_to_json(path, indent=indent)


def save_to_yaml(
    builder: DatasetBuilder,
    path: Path | str,
) -> None:
    """Save dataset to YAML file.

    Args:
        builder: DatasetBuilder to save
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "name": builder.name,
        "description": builder.description,
        "version": builder.version,
        "records": [r.model_dump(exclude_none=True) for r in builder.records],
    }

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def load_dataset(path: Path | str, format: str | None = None) -> DatasetBuilder:
    """Load dataset from file (auto-detect format).

    Args:
        path: Path to dataset file
        format: Optional format override ("json", "yaml", "yml")

    Returns:
        DatasetBuilder with loaded records

    Raises:
        ValueError: If format cannot be determined
    """
    path = Path(path)

    if format is None:
        suffix = path.suffix.lower()
        if suffix == ".json":
            format = "json"
        elif suffix in (".yaml", ".yml"):
            format = "yaml"
        else:
            raise ValueError(f"Cannot determine format from suffix '{suffix}'. Specify format explicitly.")

    if format == "json":
        return load_from_json(path)
    elif format in ("yaml", "yml"):
        return load_from_yaml(path)
    else:
        raise ValueError(f"Unknown format '{format}'. Supported: json, yaml")


__all__ = [
    "load_from_json",
    "load_from_yaml",
    "save_to_json",
    "save_to_yaml",
    "load_dataset",
]