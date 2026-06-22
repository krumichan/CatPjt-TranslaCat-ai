from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")


def chunk_list(items: list[T], size: int) -> Iterable[list[T]]:
    if size <= 0:
        raise ValueError("size must be greater than 0")

    for index in range(0, len(items), size):
        yield items[index:index + size]
