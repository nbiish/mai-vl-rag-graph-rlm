"""Qwen3-VL Embedding and Reranker integration for multimodal RAG.

This module provides local multimodal embedding and reranking capabilities
using the Qwen3-VL model family.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from vl_rag_graph_rlm.rag import SearchResult

logger = logging.getLogger("rlm.rag.qwen3vl")


@dataclass
class MultimodalDocument:
    """A multimodal document with text, image, video content and metadata."""
    id: str
    content: str  # Text content or description
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    image_path: Optional[str] = None  # Path to associated image
    video_path: Optional[str] = None  # Path to associated video


class Qwen3VLEmbeddingProvider:
    """Local multimodal embedding provider using Qwen3-VL.
    
    Supports text, images, screenshots, and videos in a unified embedding space.
    
    Example:
        >>> from vl_rag_graph_rlm.rag.qwen3vl import Qwen3VLEmbeddingProvider
        >>> 
        >>> embedder = Qwen3VLEmbeddingProvider(
        ...     model_name_or_path="Qwen/Qwen3-VL-Embedding-2B"
        ... )
        >>> 
        >>> # Embed text
        >>> text_emb = embedder.embed_text("A woman playing with her dog")
        >>> 
        >>> # Embed image
        >>> image_emb = embedder.embed_image("path/to/image.jpg")
        >>> 
        >>> # Embed multimodal input
        >>> mm_emb = embedder.embed_multimodal(
        ...     text="Describe this scene",
        ...     image="path/to/image.jpg"
        ... )
    """
    
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-VL-Embedding-2B",
        device: Optional[str] = None,
        max_length: int = 8192,
        torch_dtype: Optional[Any] = None,
        attn_implementation: Optional[str] = None,
        default_instruction: str = "Represent the user's input."
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.default_instruction = default_instruction
        
        # Import here to avoid loading at module import time
        from transformers import Qwen3VLProcessor
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLPreTrainedModel, Qwen3VLModel, Qwen3VLConfig
        from transformers.modeling_outputs import ModelOutput
        from dataclasses import dataclass
        
        @dataclass
        class Qwen3VLForEmbeddingOutput(ModelOutput):
            last_hidden_state: Optional[torch.FloatTensor] = None
            attention_mask: Optional[torch.Tensor] = None
        
        class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
            """Wrapper model for embedding generation."""
            
            def __init__(self, config):
                super().__init__(config)
                self.model = Qwen3VLModel(config)
                self.post_init()
            
            def forward(self, **kwargs):
                outputs = self.model(**kwargs)
                return Qwen3VLForEmbeddingOutput(
                    last_hidden_state=outputs.last_hidden_state,
                    attention_mask=kwargs.get('attention_mask')
                )
        
        self.Qwen3VLForEmbedding = Qwen3VLForEmbedding
        
        # Load model and processor
        logger.info(f"Loading Qwen3-VL Embedding model from {model_name_or_path}")
        
        model_kwargs = {"trust_remote_code": True}
        if torch_dtype:
            model_kwargs["torch_dtype"] = torch_dtype
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        
        self.model = Qwen3VLForEmbedding.from_pretrained(
            model_name_or_path, **model_kwargs
        ).to(self.device)
        self.processor = Qwen3VLProcessor.from_pretrained(
            model_name_or_path, padding_side='right', trust_remote_code=True
        )
        self.model.eval()
        
        # Get embedding dimension from model config
        # transformers 5.x moved hidden_size under text_config for VL models
        config = self.model.config
        if hasattr(config, 'hidden_size'):
            self.embedding_dim = config.hidden_size
        elif hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
            self.embedding_dim = config.text_config.hidden_size
        else:
            raise AttributeError(
                f"Cannot determine embedding dimension from model config: "
                f"{type(config).__name__} has neither 'hidden_size' nor 'text_config.hidden_size'"
            )
        
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    @torch.no_grad()
    def embed_text(self, text: str, instruction: Optional[str] = None) -> List[float]:
        """Generate embedding for text input."""
        inputs = [{"text": text, "instruction": instruction or self.default_instruction}]
        return self._process_inputs(inputs)[0].tolist()
    
    @torch.no_grad()
    def embed_image(
        self,
        image: Union[str, Image.Image],
        instruction: Optional[str] = None
    ) -> List[float]:
        """Generate embedding for image input.
        
        Args:
            image: Image path, URL, or PIL Image
            instruction: Optional task instruction
        """
        inputs = [{"image": image, "instruction": instruction or self.default_instruction}]
        return self._process_inputs(inputs)[0].tolist()
    
    @torch.no_grad()
    def embed_video(
        self,
        video: Union[str, List[Union[str, Image.Image]]],
        instruction: Optional[str] = None,
        fps: float = 1.0,
        max_frames: int = 64
    ) -> List[float]:
        """Generate embedding for video input.
        
        Args:
            video: Video path, URL, or list of frames
            instruction: Optional task instruction
            fps: Frame sampling rate
            max_frames: Maximum frames to sample
        """
        inputs = [{
            "video": video,
            "instruction": instruction or self.default_instruction,
            "fps": fps,
            "max_frames": max_frames
        }]
        return self._process_inputs(inputs)[0].tolist()
    
    @torch.no_grad()
    def embed_multimodal(
        self,
        text: Optional[str] = None,
        image: Optional[Union[str, Image.Image]] = None,
        video: Optional[Union[str, List[Union[str, Image.Image]]]] = None,
        instruction: Optional[str] = None,
        fps: float = 1.0,
        max_frames: int = 64
    ) -> List[float]:
        """Generate embedding for multimodal input.
        
        Args:
            text: Optional text input
            image: Optional image input
            video: Optional video input
            instruction: Optional task instruction
            fps: Frame sampling rate for video
            max_frames: Maximum frames for video
        """
        input_dict: Dict[str, Any] = {
            "instruction": instruction or self.default_instruction
        }
        if text:
            input_dict["text"] = text
        if image:
            input_dict["image"] = image
        if video:
            input_dict["video"] = video
            input_dict["fps"] = fps
            input_dict["max_frames"] = max_frames
        
        return self._process_inputs([input_dict])[0].tolist()
    
    @torch.no_grad()
    def embed_batch(
        self,
        inputs: List[Dict[str, Any]],
        batch_size: int = 8
    ) -> List[List[float]]:
        """Generate embeddings for a batch of multimodal inputs.
        
        Args:
            inputs: List of input dictionaries with text/image/video keys
            batch_size: Processing batch size
        """
        all_embeddings = []
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            embeddings = self._process_inputs(batch)
            all_embeddings.extend(embeddings.cpu().numpy().tolist())
        return all_embeddings
    
    def _process_inputs(self, inputs: List[Dict[str, Any]]) -> torch.Tensor:
        """Process inputs and generate embeddings."""
        from qwen_vl_utils.vision_process import process_vision_info
        
        # Format conversations
        conversations = []
        for inp in inputs:
            content = []
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": inp.get("instruction", self.default_instruction)}]},
                {"role": "user", "content": content}
            ]
            
            # Add video if present
            if "video" in inp:
                video = inp["video"]
                video_content = self._format_video(video, inp.get("fps", 1.0), inp.get("max_frames", 64))
                content.append({"type": "video", "video": video_content})
            
            # Add images if present
            if "image" in inp:
                image = inp["image"]
                image_content = self._format_image(image)
                content.append({"type": "image", "image": image_content})
            
            # Add text if present
            if "text" in inp:
                content.append({"type": "text", "text": inp["text"]})
            
            if not content:
                content.append({"type": "text", "text": "NULL"})
            
            conversations.append(conversation)
        
        # Process with processor
        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )
        
        try:
            images, video_inputs, video_kwargs = process_vision_info(
                conversations, image_patch_size=16,
                return_video_metadata=True, return_video_kwargs=True
            )
        except Exception as e:
            logger.error(f"Error processing vision info: {e}")
            images = None
            video_inputs = None
            video_kwargs = {'do_sample_frames': False}
        
        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos, video_metadata = None, None
        
        processed = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            do_resize=False,
            return_tensors='pt',
            **video_kwargs
        )
        
        # Move to device
        processed = {k: v.to(self.device) for k, v in processed.items()}
        
        # Forward pass
        outputs = self.model(**processed)
        
        # Pool using last token
        embeddings = self._pool_last_token(
            outputs.last_hidden_state,
            processed.get('attention_mask')
        )
        
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings
    
    def _format_video(
        self,
        video: Union[str, List[Union[str, Image.Image]]],
        fps: float,
        max_frames: int
    ) -> Union[str, List[Union[str, Image.Image]]]:
        """Format video input for processing."""
        if isinstance(video, str):
            # File path or URL
            return video if video.startswith(('http://', 'https://')) else f'file://{video}'
        elif isinstance(video, list):
            # Frame sequence
            frames = video[:max_frames] if len(video) > max_frames else video
            return [
                f'file://{f}' if isinstance(f, str) else f
                for f in frames
            ]
        return video
    
    def _format_image(self, image: Union[str, Image.Image]) -> Union[str, Image.Image]:
        """Format image input for processing."""
        if isinstance(image, str):
            return image if image.startswith(('http://', 'https://')) else f'file://{image}'
        return image
    
    def _pool_last_token(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Pool embeddings using the last valid token position."""
        if attention_mask is None:
            return hidden_state[:, -1]
        
        flipped_mask = attention_mask.flip(dims=[1])
        last_one_positions = flipped_mask.argmax(dim=1)
        col = attention_mask.shape[1] - last_one_positions - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]


class Qwen3VLRerankerProvider:
    """Local multimodal reranker using Qwen3-VL-Reranker.
    
    Performs fine-grained relevance scoring for query-document pairs
    using cross-attention mechanisms.
    
    Example:
        >>> from vl_rag_graph_rlm.rag.qwen3vl import Qwen3VLRerankerProvider
        >>> 
        >>> reranker = Qwen3VLRerankerProvider(
        ...     model_name_or_path="Qwen/Qwen3-VL-Reranker-2B"
        >>> )
        >>> 
        >>> scores = reranker.rerank(
        ...     query={"text": "What causes climate change?"},
        ...     documents=[
        ...         {"text": "Climate change is caused by greenhouse gases."},
        ...         {"image": "climate_diagram.jpg"}
        >>>     ]
        >>> )
    """
    
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-VL-Reranker-2B",
        device: Optional[str] = None,
        max_length: int = 10240,
        torch_dtype: Optional[Any] = None,
        attn_implementation: Optional[str] = None,
        default_instruction: str = "Given a search query, retrieve relevant candidates that answer the query."
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.default_instruction = default_instruction
        
        # Import here to avoid loading at module import time
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        
        logger.info(f"Loading Qwen3-VL Reranker model from {model_name_or_path}")
        
        model_kwargs = {"trust_remote_code": True}
        if torch_dtype:
            model_kwargs["torch_dtype"] = torch_dtype
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        
        # Load the full conditional generation model
        lm = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path, **model_kwargs
        ).to(self.device)
        
        self.model = lm.model  # Get the base model
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path, padding_side='left', trust_remote_code=True
        )
        self.model.eval()
        
        # Create binary classification head for yes/no scoring
        token_yes_id = self.processor.tokenizer.get_vocab()["yes"]
        token_no_id = self.processor.tokenizer.get_vocab()["no"]
        
        lm_head_weights = lm.lm_head.weight.data
        weight_yes = lm_head_weights[token_yes_id]
        weight_no = lm_head_weights[token_no_id]
        
        D = weight_yes.size(0)
        self.score_linear = torch.nn.Linear(D, 1, bias=False)
        with torch.no_grad():
            self.score_linear.weight[0] = weight_yes - weight_no
        
        self.score_linear.eval()
        self.score_linear.to(self.device).to(self.model.dtype)
        
        logger.info("Reranker model loaded successfully")
    
    @torch.no_grad()
    def rerank(
        self,
        query: Dict[str, Any],
        documents: List[Dict[str, Any]],
        instruction: Optional[str] = None,
        fps: float = 1.0,
        max_frames: int = 64
    ) -> List[Tuple[int, float]]:
        """Rerank documents based on query relevance.
        
        Args:
            query: Query dict with text/image/video keys
            documents: List of document dicts with text/image/video keys
            instruction: Optional task instruction
            fps: Frame sampling rate for video
            max_frames: Maximum frames for video
            
        Returns:
            List of (document_index, relevance_score) tuples, sorted by score
        """
        scores = []
        
        for i, doc in enumerate(documents):
            score = self._score_pair(
                query, doc,
                instruction=instruction or self.default_instruction,
                fps=fps,
                max_frames=max_frames
            )
            scores.append((i, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def _score_pair(
        self,
        query: Dict[str, Any],
        document: Dict[str, Any],
        instruction: str,
        fps: float,
        max_frames: int
    ) -> float:
        """Score a single query-document pair."""
        from qwen_vl_utils.vision_process import process_vision_info
        
        # Format conversation
        conversation = [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
                }]
            },
            {
                "role": "user",
                "content": self._format_content(query, document, instruction, fps, max_frames)
            }
        ]
        
        # Process
        text = self.processor.apply_chat_template(
            [conversation], tokenize=False, add_generation_prompt=True
        )
        
        try:
            images, videos, video_kwargs = process_vision_info(
                [conversation],
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True
            )
        except Exception as e:
            logger.error(f"Error processing vision info: {e}")
            return 0.0
        
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None
        
        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            truncation=False,
            padding=False,
            do_resize=False,
            **video_kwargs
        )
        
        # Truncate while preserving special tokens
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self._truncate_tokens(
                inputs['input_ids'][i][:-5],
                self.max_length,
                self.processor.tokenizer.all_special_ids
            ) + inputs['input_ids'][i][-5:]
        
        # Pad
        temp_inputs = self.processor.tokenizer.pad(
            {'input_ids': inputs['input_ids']},
            padding=True,
            return_tensors="pt",
            max_length=self.max_length
        )
        for key in temp_inputs:
            inputs[key] = temp_inputs[key]
        
        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        # Forward pass
        outputs = self.model(**inputs)
        last_hidden = outputs.last_hidden_state[:, -1]
        
        # Score
        score = self.score_linear(last_hidden)
        score = torch.sigmoid(score).squeeze(-1).item()
        
        return score
    
    def _format_content(
        self,
        query: Dict[str, Any],
        document: Dict[str, Any],
        instruction: str,
        fps: float,
        max_frames: int
    ) -> List[Dict[str, Any]]:
        """Format query and document into conversation content."""
        content = []
        
        # Add instruction
        content.append({"type": "text", "text": f"<Instruct>: {instruction}"})
        
        # Add query with prefix
        query_content = [{"type": "text", "text": "<Query>:"}]
        query_content.extend(self._format_multimodal_input(query, fps, max_frames))
        content.extend(query_content)
        
        # Add document with prefix
        doc_content = [{"type": "text", "text": "\n<Document>:"}]
        doc_content.extend(self._format_multimodal_input(document, fps, max_frames))
        content.extend(doc_content)
        
        return content
    
    def _format_multimodal_input(
        self,
        input_dict: Dict[str, Any],
        fps: float,
        max_frames: int
    ) -> List[Dict[str, Any]]:
        """Format multimodal input into content items."""
        content = []
        
        # Add video if present
        if "video" in input_dict:
            video = input_dict["video"]
            if isinstance(video, str):
                video_path = video if video.startswith(('http://', 'https://')) else f'file://{video}'
                content.append({"type": "video", "video": video_path})
            elif isinstance(video, list):
                frames = video[:max_frames] if len(video) > max_frames else video
                frame_paths = [
                    f'file://{f}' if isinstance(f, str) else f
                    for f in frames
                ]
                content.append({"type": "video", "video": frame_paths})
        
        # Add image if present
        if "image" in input_dict:
            image = input_dict["image"]
            if isinstance(image, str):
                image_path = image if image.startswith(('http://', 'https://')) else f'file://{image}'
                content.append({"type": "image", "image": image_path})
            else:
                content.append({"type": "image", "image": image})
        
        # Add text if present
        if "text" in input_dict:
            content.append({"type": "text", "text": input_dict["text"]})
        
        return content
    
    def _truncate_tokens(
        self,
        tokens: List[int],
        max_length: int,
        special_tokens: List[int]
    ) -> List[int]:
        """Truncate tokens while preserving special tokens."""
        if len(tokens) <= max_length:
            return tokens
        
        special_set = set(special_tokens)
        num_special = sum(1 for t in tokens if t in special_set)
        num_non_special_to_keep = max_length - num_special
        
        result = []
        non_special_count = 0
        for token in tokens:
            if token in special_set:
                result.append(token)
            elif non_special_count < num_non_special_to_keep:
                result.append(token)
                non_special_count += 1
        
        return result


# Convenience factory functions
def create_qwen3vl_embedder(
    model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
    **kwargs
) -> Qwen3VLEmbeddingProvider:
    """Create a Qwen3-VL embedding provider."""
    return Qwen3VLEmbeddingProvider(
        model_name_or_path=model_name,
        **kwargs
    )


def create_qwen3vl_reranker(
    model_name: str = "Qwen/Qwen3-VL-Reranker-2B",
    **kwargs
) -> Qwen3VLRerankerProvider:
    """Create a Qwen3-VL reranker provider."""
    return Qwen3VLRerankerProvider(
        model_name_or_path=model_name,
        **kwargs
    )
