"""Dataset builder for VideoDeepSearch evaluation.

Provides API for constructing and managing evaluation datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from videodeepsearch.evaluation.datasets.schema import (
    EvalRecord,
    EvalInput,
    EvalExpectation,
    ExpectedFact,
)

class DatasetBuilder:
    """Build evaluation datasets for VideoDeepSearch agent.

    Usage:
        builder = DatasetBuilder("my_eval_dataset")

        builder.add_record(
            user_id="user1",
            video_ids=["video1", "video2"],
            user_demand="Find information about X",
            expected_response="The answer is...",
        )

        # Save to JSON
        builder.save_to_json("eval_dataset.json")

        # Load from JSON
        builder2 = DatasetBuilder("loaded")
        builder2.load_from_json("eval_dataset.json")
    """

    def __init__(
        self,
        name: str,
        description: str | None = None,
        version: str | None = None,
    ):
        self.name = name
        self.description = description
        self.version = version
        self.records: list[EvalRecord] = []

    def add_record(
        self,
        user_id: str,
        ground_truth_video_ids: list[str],
        total_video_haystack_ids: list[str],
        user_demand: str,
        session_id: str | None = None,
        expected_response: str | None = None,
        expected_facts: list[dict[str, str] | ExpectedFact] | None = None,
        expected_video_ids: list[str] | None = None,
        notes: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvalRecord:
        """Add a single evaluation record.

        Args:
            user_id: User identifier
            video_ids: List of video IDs to search
            user_demand: User's question or request
            session_id: Optional session ID (auto-generated if None)
            expected_response: Expected response text
            expected_facts: List of expected facts (dict or ExpectedFact)
            expected_video_ids: Expected related video IDs
            notes: Notes for evaluation reference
            guidelines: Custom guidelines for this record
            metadata: Additional context

        Returns:
            The created EvalRecord
        """

        inputs = EvalInput(
            user_id=user_id,
            ground_truth_video_ids=ground_truth_video_ids,
            total_video_haystack_ids=total_video_haystack_ids,
            user_demand=user_demand,
            session_id=session_id, #type:ignore
        )

        
        exp_facts = None
        if expected_facts:
            exp_facts = [
                ExpectedFact(**f) if isinstance(f, dict) else f
                for f in expected_facts
            ]

        expectations = EvalExpectation(
            expected_response=expected_response,
            expected_facts=exp_facts,
            expected_video_ids=expected_video_ids,
            notes=notes,
        )


        record = EvalRecord(
            inputs=inputs,
            expectations=expectations,
            metadata=metadata,
        )
        self.records.append(record)
        return record

    def add_record_from_dict(self, data: dict[str, Any]) -> EvalRecord:
        record = EvalRecord.from_dict(data)
        self.records.append(record)
        return record

    def add_records_from_list(self, records: list[dict[str, Any]]) -> list[EvalRecord]:
        """Add multiple records from a list of dictionaries."""
        return [self.add_record_from_dict(r) for r in records]

    def load_from_json(self, path: Path | str) -> list[EvalRecord]:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            records_data = data
        elif "records" in data:
            self.name = data.get("name", self.name)
            self.description = data.get("description", self.description)
            self.version = data.get("version", self.version)
            records_data = data["records"]
        else:
            records_data = [data]

        return self.add_records_from_list(records_data)

    def save_to_json(self, path: Path | str, indent: int = 2) -> None:
        """Save dataset to a JSON file.

        Args:
            path: Output file path
            indent: JSON indentation (default: 2)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "records": [r.model_dump(exclude_none=True) for r in self.records],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

    def get_records(self) -> list[EvalRecord]:
        """Get all records as a copy."""
        return self.records.copy()

    def to_mlflow_format(self) -> list[dict[str, Any]]:
        """Convert all records to MLflow dataset format.

        Returns:
            List of dicts in MLflow format
        """
        return [r.to_mlflow_format() for r in self.records]

    def create_mlflow_dataset(
        self,
        experiment_id: str | None = None,
        tags: dict[str, Any] | None = None
    ) -> Any:
        """Create an MLflow GenAI dataset from records.

        Args:
            experiment_id: MLflow experiment ID (optional)

        Returns:
            MLflow Dataset object
        """
        import mlflow

        dataset = mlflow.genai.datasets.create_dataset( #type:ignore
            name=self.name,
            experiment_id=experiment_id,
            tags=tags
        )

        records_mlflow = self.to_mlflow_format()
        if records_mlflow:
            dataset.merge_records(records_mlflow)

        return dataset

    def __len__(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        return f"DatasetBuilder(name={self.name!r}, records={len(self.records)})"

    def __iter__(self):
        return iter(self.records)
