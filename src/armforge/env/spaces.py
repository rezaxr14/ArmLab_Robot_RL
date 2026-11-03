"""Minimal space description. ArmForge deliberately has no Gymnasium dependency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Box:
    low: np.ndarray
    high: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        return self.low.shape

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high).astype(np.float32)

    def contains(self, x) -> bool:
        x = np.asarray(x)
        return x.shape == self.shape and bool(np.all(x >= self.low) and np.all(x <= self.high))

    @staticmethod
    def symmetric(dim: int, bound: float = 1.0) -> Box:
        return Box(np.full(dim, -bound, np.float32), np.full(dim, bound, np.float32))
