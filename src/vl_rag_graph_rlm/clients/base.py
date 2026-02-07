"""Base class for all Language Model clients."""

from abc import ABC, abstractmethod
from typing import Any

from vl_rag_graph_rlm.types import ModelUsageSummary, UsageSummary


class BaseLM(ABC):
    """
    Base class for all language model clients.
    Provides a unified interface for different providers.
    """

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    @abstractmethod
    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        """Make a synchronous completion call."""
        raise NotImplementedError

    @abstractmethod
    async def acompletion(self, prompt: str | list[dict[str, Any]], model: str | None = None, **kwargs) -> str:
        """Make an asynchronous completion call."""
        raise NotImplementedError

    @abstractmethod
    def get_usage_summary(self) -> UsageSummary:
        """Get aggregated usage summary for all model calls."""
        raise NotImplementedError

    @abstractmethod
    def get_last_usage(self) -> ModelUsageSummary:
        """Get usage for the last model call."""
        raise NotImplementedError
