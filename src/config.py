"""Explicit, serializable experiment settings. Credentials never belong here."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 4
    candidate_k: int = 15
    rerank: bool = True
    chunk_tokens: int = 240  # Includes headings and tokenizer special tokens.
    overlap_tokens: int = 24
    batch_size: int = 32

    def __post_init__(self):
        if self.top_k < 1 or self.candidate_k < self.top_k:
            raise ValueError("Require candidate_k >= top_k >= 1")
        if self.chunk_tokens < 8 or not 0 <= self.overlap_tokens < self.chunk_tokens:
            raise ValueError("Invalid chunk/overlap token budget")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")


@dataclass(frozen=True)
class GenerationConfig:
    model: str = "qwen/qwen3.8-27b"
    temperature: float = 0.0
    max_output_tokens: int = 2400
    context_tokens: int = 3000
    counter_encoding: str = "cl100k_base"
    # This counter is a proxy, not the provider's model-specific tokenizer.
    max_history_turns: int = 6

    def __post_init__(self):
        if self.max_output_tokens < 1 or self.context_tokens < 1:
            raise ValueError("Generation budgets must be positive")
        if self.max_history_turns < 0 or not 0 <= self.temperature <= 2:
            raise ValueError("Invalid history limit or temperature")
