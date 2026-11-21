from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HumanEvaluation:
    correctness: int
    completeness: int
    relevance: int
    clarity: int
    instruction_following: int
    notes: str = ""

    def __post_init__(self) -> None:
        if any(not 0 <= value <= 2 for value in self.__dict__.values() if isinstance(value, int)):
            raise ValueError("Human ratings must be integers from 0 to 2")

    @property
    def total(self) -> int:
        return self.correctness + self.completeness + self.relevance + self.clarity + self.instruction_following
