"""datasets/__init__.py — Dataset adapters and fusion pipeline."""

from datasets.normaliser import RawTransition, UniversalNormaliser
from datasets.preloader import DatasetPreloader

__all__ = [
    "RawTransition",
    "UniversalNormaliser",
    "DatasetPreloader",
]
