"""Procedural kidney collecting-system mesh generator."""

from .config import GeneratorConfig
from .generator import generate_case

__all__ = ["GeneratorConfig", "generate_case"]
__version__ = "0.7.0"
