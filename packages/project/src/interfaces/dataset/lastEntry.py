from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass
class LastEntry[T]:
    index: int
    data: T
