"""
Ragify - Intelligent Context Orchestration Plugin

A generic, open-source plugin for intelligent context management in LLM-powered applications.
"""

__version__ = "0.1.0"
__author__ = "Sumit Nemade"
__email__ = "nemadesumit@gmail.com"

from .core import ContextOrchestrator
from .models import Context, ContextSource, RelevanceScore, PrivacyLevel
from .exceptions import ICOException, ContextNotFoundError, SourceConnectionError

__all__ = [
    "ContextOrchestrator",
    "Context",
    "ContextSource", 
    "RelevanceScore",
    "PrivacyLevel",
    "ICOException",
    "ContextNotFoundError",
    "SourceConnectionError",
]
