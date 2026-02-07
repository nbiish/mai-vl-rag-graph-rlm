"""Type definitions for VL_RAG_GRAPH_RLM."""

from dataclasses import dataclass
from typing import Any, Literal, TypedDict, Optional, Callable, Awaitable

# Provider types
ProviderType = Literal[
    "openai",
    "openai_compatible",
    "azure_openai",
    "openrouter",
    "zenmux",
    "zai",
    "anthropic",
    "anthropic_compatible",
    "gemini",
    "groq",
    "mistral",
    "fireworks",
    "together",
    "deepseek",
    "sambanova",
    "nebius",
    "cerebras",
    "litellm",
]

EnvironmentType = Literal["local", "docker", "modal"]


########################################################
########    Types for LM Usage Tracking      ###########
########################################################

@dataclass
class ModelUsageSummary:
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelUsageSummary":
        return cls(
            total_calls=data.get("total_calls", 0),
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
        )


@dataclass
class UsageSummary:
    model_usage_summaries: dict[str, ModelUsageSummary]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_usage_summaries": {
                model: usage_summary.to_dict()
                for model, usage_summary in self.model_usage_summaries.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UsageSummary":
        return cls(
            model_usage_summaries={
                model: ModelUsageSummary.from_dict(usage_summary)
                for model, usage_summary in data.get("model_usage_summaries", {}).items()
            },
        )


########################################################
########   Types for REPL and RLM Results    ###########
########################################################

@dataclass
class VLRAGGraphRLMChatCompletion:
    """Record of a single LLM call."""

    provider: str
    model: str
    prompt: str | dict[str, Any]
    response: str
    usage_summary: UsageSummary
    execution_time: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt": self.prompt,
            "response": self.response,
            "usage_summary": self.usage_summary.to_dict(),
            "execution_time": self.execution_time,
        }


@dataclass
class REPLResult:
    stdout: str
    stderr: str
    locals: dict
    execution_time: float
    llm_calls: list["VLRAGGraphRLMChatCompletion"]

    def __init__(
        self,
        stdout: str,
        stderr: str,
        locals: dict,
        execution_time: float = 0.0,
        vlrag_calls: list["VLRAGGraphRLMChatCompletion"] | None = None,
    ):
        self.stdout = stdout
        self.stderr = stderr
        self.locals = locals
        self.execution_time = execution_time
        self.vlrag_calls = vlrag_calls or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "locals": {k: str(v) for k, v in self.locals.items()},
            "execution_time": self.execution_time,
            "vlrag_calls": [call.to_dict() for call in self.vlrag_calls],
        }


@dataclass
class CodeBlock:
    code: str
    result: REPLResult

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "result": self.result.to_dict()}


@dataclass
class RLMIteration:
    prompt: str | dict[str, Any]
    response: str
    code_blocks: list[CodeBlock]
    final_answer: str | None = None
    iteration_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "code_blocks": [block.to_dict() for block in self.code_blocks],
            "final_answer": self.final_answer,
            "iteration_time": self.iteration_time,
        }


########################################################
########   Configuration Types               ###########
########################################################

class Message(TypedDict):
    """LLM message format."""

    role: str
    content: str


class ProviderConfig(TypedDict, total=False):
    """Configuration for a provider."""

    provider: ProviderType
    api_key: str
    api_base: str
    model: str
    timeout: int
    max_retries: int


class RLMConfig(TypedDict, total=False):
    """Configuration for RLM instance."""

    provider: ProviderType
    model: str
    recursive_model: Optional[str]
    api_key: str
    api_base: str
    max_depth: int
    max_iterations: int
    temperature: float
    timeout: int


class REPLEnvironment(TypedDict, total=False):
    """REPL execution environment."""

    context: str
    query: str
    recursive_llm: Callable[[str, str], str]
    re: Any
