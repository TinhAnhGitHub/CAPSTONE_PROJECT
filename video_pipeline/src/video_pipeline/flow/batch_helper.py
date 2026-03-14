"""Helper functions for batch processing in flows."""


def make_batches(items: list, batch_size: int) -> list[list]:
    """Split items into batches of specified size."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
