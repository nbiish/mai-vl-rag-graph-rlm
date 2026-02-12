"""Token usage tracking and cost estimation for VL-RAG-Graph-RLM.

Tracks API token consumption and estimates costs across different providers.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


# Cost per 1K tokens (USD) — approximate rates as of early 2025
_PROVIDER_COSTS = {
    "openai": {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    },
    "anthropic": {
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-5-haiku": {"input": 0.0008, "output": 0.004},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
    },
    "groq": {
        "default": {"input": 0.0005, "output": 0.0005},  # Very low cost
    },
    "deepseek": {
        "deepseek-chat": {"input": 0.00014, "output": 0.00028},
        "deepseek-reasoner": {"input": 0.00014, "output": 0.00219},
    },
    "mistral": {
        "mistral-large": {"input": 0.002, "output": 0.006},
        "mistral-medium": {"input": 0.0006, "output": 0.0018},
    },
    "gemini": {
        "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
        "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    },
    "sambanova": {
        "default": {"input": 0.001, "output": 0.002},
    },
    "nebius": {
        "default": {"input": 0.0002, "output": 0.0006},
    },
    "openrouter": {
        "default": {"input": 0.001, "output": 0.002},
    },
    "cerebras": {
        "default": {"input": 0.0005, "output": 0.0005},
    },
    "fireworks": {
        "default": {"input": 0.0005, "output": 0.0005},
    },
    "together": {
        "default": {"input": 0.0006, "output": 0.0006},
    },
    "ollama": {
        "default": {"input": 0, "output": 0},  # Free — local inference
    },
}


def _get_model_costs(provider: str, model: str) -> Dict[str, float]:
    """Get cost rates for a provider/model."""
    provider_rates = _PROVIDER_COSTS.get(provider, {})
    
    # Try exact model match first
    if model in provider_rates:
        return provider_rates[model]
    
    # Try partial match
    for model_key, rates in provider_rates.items():
        if model_key in model.lower():
            return rates
    
    # Fall back to default
    return provider_rates.get("default", {"input": 0.001, "output": 0.002})


@dataclass
class TokenUsage:
    """Token usage for a single API call."""
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
    
    @property
    def cost_usd(self) -> float:
        """Calculate estimated cost in USD."""
        rates = _get_model_costs(self.provider, self.model)
        input_cost = (self.input_tokens / 1000) * rates["input"]
        output_cost = (self.output_tokens / 1000) * rates["output"]
        return round(input_cost + output_cost, 6)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "timestamp": self.timestamp,
        }


class TokenTracker:
    """Track token usage across multiple API calls."""
    
    def __init__(self):
        self.usages: List[TokenUsage] = []
    
    def record(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> TokenUsage:
        """Record a new usage entry."""
        usage = TokenUsage(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self.usages.append(usage)
        return usage
    
    @property
    def total_input_tokens(self) -> int:
        return sum(u.input_tokens for u in self.usages)
    
    @property
    def total_output_tokens(self) -> int:
        return sum(u.output_tokens for u in self.usages)
    
    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens
    
    @property
    def total_cost_usd(self) -> float:
        return round(sum(u.cost_usd for u in self.usages), 6)
    
    def get_by_provider(self, provider: str) -> List[TokenUsage]:
        """Get all usages for a specific provider."""
        return [u for u in self.usages if u.provider == provider]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.usages:
            return {
                "total_calls": 0,
                "total_tokens": 0,
                "total_cost_usd": 0,
                "by_provider": {},
            }
        
        by_provider = {}
        for usage in self.usages:
            p = usage.provider
            if p not in by_provider:
                by_provider[p] = {"tokens": 0, "cost": 0, "calls": 0}
            by_provider[p]["tokens"] += usage.total_tokens
            by_provider[p]["cost"] += usage.cost_usd
            by_provider[p]["calls"] += 1
        
        return {
            "total_calls": len(self.usages),
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "by_provider": by_provider,
        }
    
    def print_report(self) -> None:
        """Print a formatted usage report."""
        if not self.usages:
            print("No token usage recorded.")
            return
        
        summary = self.get_summary()
        print("\n" + "=" * 50)
        print("TOKEN USAGE REPORT")
        print("=" * 50)
        print(f"Total API calls: {summary['total_calls']}")
        print(f"Total tokens: {summary['total_tokens']:,}")
        print(f"  Input:  {self.total_input_tokens:,}")
        print(f"  Output: {self.total_output_tokens:,}")
        print(f"Estimated cost: ${summary['total_cost_usd']:.4f}")
        
        if summary['by_provider']:
            print("\nBy provider:")
            for provider, stats in summary['by_provider'].items():
                print(f"  {provider}: {stats['calls']} calls, {stats['tokens']:,} tokens, ${stats['cost']:.4f}")
        
        print("=" * 50)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export all data as dictionary."""
        return {
            "usages": [u.to_dict() for u in self.usages],
            "summary": self.get_summary(),
        }


# Global tracker instance
default_tracker = TokenTracker()


def estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    """Rough token estimate from character count.
    
    Very approximate — actual tokenization varies by model.
    """
    return len(text) // chars_per_token
