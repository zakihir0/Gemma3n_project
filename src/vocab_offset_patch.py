"""
Vocabulary Offset Patch for Gemma3n Models

This module provides patches to fix vocab offset issues when extending text vocabulary:
1. Shifts audio_vocab_offset and vision_vocab_offset after text_vocab_size expansion
2. Ensures proper offset references in embedding modules
3. Patches forward pass token ID judgment conditions
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class VocabOffsetConfig:
    """Configuration for vocabulary offsets"""
    
    def __init__(self, 
                 original_text_vocab_size: int,
                 new_text_vocab_size: int,
                 audio_vocab_size: int = 1024,
                 vision_vocab_size: int = 1024):
        
        self.original_text_vocab_size = original_text_vocab_size
        self.new_text_vocab_size = new_text_vocab_size
        self.audio_vocab_size = audio_vocab_size
        self.vision_vocab_size = vision_vocab_size
        
        # Calculate shifted offsets
        self.text_vocab_offset = 0
        self.audio_vocab_offset = new_text_vocab_size
        self.vision_vocab_offset = new_text_vocab_size + audio_vocab_size
        
        # Total vocabulary size
        self.total_vocab_size = new_text_vocab_size + audio_vocab_size + vision_vocab_size
        
        logger.info(f"Vocabulary offset configuration:")
        logger.info(f"  Text vocab: {self.text_vocab_offset} - {self.audio_vocab_offset - 1}")
        logger.info(f"  Audio vocab: {self.audio_vocab_offset} - {self.vision_vocab_offset - 1}")
        logger.info(f"  Vision vocab: {self.vision_vocab_offset} - {self.total_vocab_size - 1}")


class PatchedEmbedding(nn.Module):
    """Patched embedding module with proper offset handling"""
    
    def __init__(self, 
                 original_embedding: nn.Embedding,
                 vocab_config: VocabOffsetConfig,
                 modality: str = "text"):
        super().__init__()
        
        self.vocab_config = vocab_config
        self.modality = modality
        self.original_embedding = original_embedding
        
        # Create separate embeddings for each modality
        if modality == "text":
            self.embedding = nn.Embedding(
                vocab_config.new_text_vocab_size,
                original_embedding.embedding_dim,
                padding_idx=original_embedding.padding_idx
            )
            # Copy original weights for existing text tokens
            with torch.no_grad():
                self.embedding.weight[:vocab_config.original_text_vocab_size] = \
                    original_embedding.weight[:vocab_config.original_text_vocab_size]
                    
        elif modality == "audio":
            self.embedding = nn.Embedding(
                vocab_config.audio_vocab_size,
                original_embedding.embedding_dim
            )
            
        elif modality == "vision":
            self.embedding = nn.Embedding(
                vocab_config.vision_vocab_size,
                original_embedding.embedding_dim
            )
            
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass with proper offset handling"""
        
        if self.modality == "text":
            # Handle text tokens (0 to text_vocab_size-1)
            mask = (input_ids >= 0) & (input_ids < self.vocab_config.new_text_vocab_size)
            valid_ids = torch.where(mask, input_ids, 0)
            return self.embedding(valid_ids) * mask.unsqueeze(-1).float()
            
        elif self.modality == "audio":
            # Handle audio tokens (audio_vocab_offset to vision_vocab_offset-1)
            mask = (input_ids >= self.vocab_config.audio_vocab_offset) & \
                   (input_ids < self.vocab_config.vision_vocab_offset)
            # Shift to 0-based indexing for audio embedding
            shifted_ids = input_ids - self.vocab_config.audio_vocab_offset
            valid_ids = torch.where(mask, shifted_ids, 0)
            return self.embedding(valid_ids) * mask.unsqueeze(-1).float()
            
        elif self.modality == "vision":
            # Handle vision tokens (vision_vocab_offset to total_vocab_size-1)
            mask = (input_ids >= self.vocab_config.vision_vocab_offset) & \
                   (input_ids < self.vocab_config.total_vocab_size)
            # Shift to 0-based indexing for vision embedding
            shifted_ids = input_ids - self.vocab_config.vision_vocab_offset
            valid_ids = torch.where(mask, shifted_ids, 0)
            return self.embedding(valid_ids) * mask.unsqueeze(-1).float()
            
        else:
            raise ValueError(f"Unknown modality: {self.modality}")


class VocabOffsetPatcher:
    """Main patcher class to apply vocabulary offset fixes"""
    
    def __init__(self, vocab_config: VocabOffsetConfig):
        self.vocab_config = vocab_config
        
    def patch_model(self, model: nn.Module) -> nn.Module:
        """Apply patches to the model"""
        
        logger.info("Applying vocabulary offset patches...")
        
        # Patch embedding layers
        if hasattr(model, 'embed_tokens'):
            model.embed_tokens = PatchedEmbedding(
                model.embed_tokens, 
                self.vocab_config, 
                "text"
            )
            
        if hasattr(model, 'embed_audio'):
            model.embed_audio = PatchedEmbedding(
                model.embed_audio, 
                self.vocab_config, 
                "audio"
            )
            
        if hasattr(model, 'embed_vision'):
            model.embed_vision = PatchedEmbedding(
                model.embed_vision, 
                self.vocab_config, 
                "vision"
            )
            
        # Patch forward method if needed
        self._patch_forward_method(model)
        
        logger.info("Vocabulary offset patches applied successfully")
        return model
        
    def _patch_forward_method(self, model: nn.Module):
        """Patch the forward method to handle token ID ranges properly"""
        
        original_forward = model.forward
        vocab_config = self.vocab_config
        
        def patched_forward(input_ids: torch.Tensor, **kwargs):
            """Patched forward with strict token ID range checking"""
            
            # Validate token IDs are within expected ranges
            max_id = torch.max(input_ids)
            if max_id >= vocab_config.total_vocab_size:
                logger.warning(f"Found token ID {max_id} >= total_vocab_size {vocab_config.total_vocab_size}")
                # Clamp to valid range
                input_ids = torch.clamp(input_ids, 0, vocab_config.total_vocab_size - 1)
                
            return original_forward(input_ids, **kwargs)
            
        model.forward = patched_forward
        
    def create_token_type_mask(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create masks for different token types"""
        
        text_mask = (input_ids >= 0) & (input_ids < self.vocab_config.new_text_vocab_size)
        
        audio_mask = (input_ids >= self.vocab_config.audio_vocab_offset) & \
                     (input_ids < self.vocab_config.vision_vocab_offset)
                     
        vision_mask = (input_ids >= self.vocab_config.vision_vocab_offset) & \
                      (input_ids < self.vocab_config.total_vocab_size)
                      
        return {
            "text": text_mask,
            "audio": audio_mask,
            "vision": vision_mask
        }


def apply_vocab_offset_patch(model: nn.Module, 
                           original_text_vocab_size: int,
                           new_text_vocab_size: int,
                           audio_vocab_size: int = 1024,
                           vision_vocab_size: int = 1024) -> nn.Module:
    """Convenience function to apply vocab offset patches"""
    
    vocab_config = VocabOffsetConfig(
        original_text_vocab_size=original_text_vocab_size,
        new_text_vocab_size=new_text_vocab_size,
        audio_vocab_size=audio_vocab_size,
        vision_vocab_size=vision_vocab_size
    )
    
    patcher = VocabOffsetPatcher(vocab_config)
    return patcher.patch_model(model)


def get_token_type(token_id: int, vocab_config: VocabOffsetConfig) -> str:
    """Determine token type based on ID and vocab config"""
    
    if 0 <= token_id < vocab_config.new_text_vocab_size:
        return "text"
    elif vocab_config.audio_vocab_offset <= token_id < vocab_config.vision_vocab_offset:
        return "audio"
    elif vocab_config.vision_vocab_offset <= token_id < vocab_config.total_vocab_size:
        return "vision"
    else:
        return "unknown"