from __future__ import annotations

from typing import ClassVar, TypeVar

T = TypeVar("T")


class StageRegistry:
    """Registers task classes as pipeline stages via a class decorator.

    Usage
    -----
    @StageRegistry.register
    class VideoRegistryTask(BaseTask[...]):
        ...

    Then in the flow:
        await tracker.start_video(video_id, StageRegistry.all_stage_names())
        await tracker.complete_stage(video_id, VideoRegistryTask.__name__)
    """

    _registry: ClassVar[list[str]] = []

    @classmethod
    def register(cls, task_class: type[T]) -> type[T]:
        cls._registry.append(task_class.__name__)
        return task_class

    @classmethod
    def all_stage_names(cls) -> list[str]:
        return list(cls._registry)
