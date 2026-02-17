"""Streaming output support for RLM responses.

Provides real-time token streaming for LLM responses to improve
perceived latency and user experience.
"""

import asyncio
import sys
from typing import AsyncIterator, Optional, Callable


class StreamingResponseHandler:
    """Handle streaming LLM responses with real-time output.
    
    Example:
        >>> handler = StreamingResponseHandler()
        >>> async for token in stream_llm_response(prompt):
        ...     handler.handle_token(token)
        >>> full_response = handler.get_full_response()
    """
    
    def __init__(
        self,
        stream_to_stdout: bool = True,
        buffer_size: int = 1,
        on_token: Optional[Callable[[str], None]] = None
    ):
        """Initialize streaming handler.
        
        Args:
            stream_to_stdout: Print tokens to stdout as they arrive
            buffer_size: Minimum tokens before flushing (1 = immediate)
            on_token: Optional callback for each token
        """
        self.stream_to_stdout = stream_to_stdout
        self.buffer_size = buffer_size
        self.on_token = on_token
        self._buffer = []
        self._full_response = []
        self._token_count = 0
    
    def handle_token(self, token: str) -> None:
        """Process a single token from the stream."""
        self._full_response.append(token)
        self._token_count += 1
        
        if self.stream_to_stdout:
            self._buffer.append(token)
            if len(self._buffer) >= self.buffer_size:
                self._flush_buffer()
        
        if self.on_token:
            self.on_token(token)
    
    def _flush_buffer(self) -> None:
        """Flush buffered tokens to stdout."""
        if self._buffer:
            text = "".join(self._buffer)
            sys.stdout.write(text)
            sys.stdout.flush()
            self._buffer = []
    
    def finalize(self) -> str:
        """Finalize streaming and return full response."""
        self._flush_buffer()
        if self.stream_to_stdout:
            sys.stdout.write("\n")
            sys.stdout.flush()
        return "".join(self._full_response)
    
    def get_full_response(self) -> str:
        """Get the complete response received so far."""
        return "".join(self._full_response)
    
    @property
    def token_count(self) -> int:
        """Number of tokens received."""
        return self._token_count


async def stream_rlm_response(
    client,
    messages: list,
    model: Optional[str] = None,
    temperature: float = 0.0,
    stream_handler: Optional[StreamingResponseHandler] = None,
    **kwargs
) -> str:
    """Execute RLM with streaming output.
    
    Args:
        client: LLM client with streaming support
        messages: Message list for the conversation
        model: Optional model override
        temperature: Sampling temperature
        stream_handler: Optional custom handler
        **kwargs: Additional arguments for the client
        
    Returns:
        Full response text
    """
    handler = stream_handler or StreamingResponseHandler()
    
    try:
        # Check if client supports streaming
        if hasattr(client, 'acompletion_stream'):
            # Use native streaming
            async for token in client.acompletion_stream(
                messages,
                model=model,
                temperature=temperature,
                **kwargs
            ):
                handler.handle_token(token)
        elif hasattr(client, 'acompletion'):
            # Fallback: simulate streaming by word
            response = await client.acompletion(
                messages,
                model=model,
                temperature=temperature,
                **kwargs
            )
            # Stream word by word
            words = response.split(" ")
            for word in words:
                handler.handle_token(word + " ")
                # Small delay for visual effect
                await asyncio.sleep(0.01)
        else:
            raise ValueError("Client does not support completion methods")
        
        return handler.finalize()
        
    except Exception as e:
        # Ensure buffer is flushed even on error
        handler._flush_buffer()
        raise e


def enable_streaming_for_rlm(rlm_instance) -> None:
    """Enable streaming mode for an RLM instance.
    
    Patches the RLM instance to use streaming for completions.
    
    Args:
        rlm_instance: VLRAGGraphRLM instance to patch
    """
    original_acompletion = rlm_instance.client.acompletion
    
    async def streaming_acompletion(messages, **kwargs):
        """Wrapped completion with streaming output."""
        handler = StreamingResponseHandler(stream_to_stdout=True)
        
        # Check if streaming is supported
        if hasattr(rlm_instance.client, 'acompletion_stream'):
            async for token in rlm_instance.client.acompletion_stream(messages, **kwargs):
                handler.handle_token(token)
            return handler.finalize()
        else:
            # Non-streaming fallback
            response = await original_acompletion(messages, **kwargs)
            return response
    
    # Replace the method
    rlm_instance.client.acompletion = streaming_acompletion


class StreamingReportGenerator:
    """Generate reports with streaming progress updates."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.sections = []
    
    def start_section(self, title: str) -> None:
        """Start a new report section."""
        self.sections.append(title)
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"{title}")
            print(f"{'='*60}")
            sys.stdout.flush()
    
    def add_chunk(self, content: str, source: str = "") -> None:
        """Add a content chunk with source attribution."""
        if self.verbose:
            if source:
                print(f"\n[Source: {source}]")
            print(content)
            sys.stdout.flush()
    
    def add_progress(self, message: str) -> None:
        """Add a progress update."""
        if self.verbose:
            print(f"  → {message}")
            sys.stdout.flush()
    
    def finalize(self) -> dict:
        """Finalize and return report metadata."""
        return {
            "sections": self.sections,
            "completed": True
        }
