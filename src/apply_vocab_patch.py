"""
Apply Vocabulary Offset Patches to Gemma3n Models

This script integrates the vocab offset patches into the existing training pipeline.
"""

import torch
import logging
from typing import Optional, Dict, Any

from vocab_offset_patch import apply_vocab_offset_patch, VocabOffsetConfig
from config_updater import ConfigUpdater, create_default_config

logger = logging.getLogger(__name__)


def patch_gemma3n_model(model,
                       tokenizer,
                       original_text_vocab_size: Optional[int] = None,
                       new_text_vocab_size: Optional[int] = None,
                       audio_vocab_size: int = 1024,
                       vision_vocab_size: int = 1024):
    """
    Apply vocabulary offset patches to a Gemma3n model
    
    Args:
        model: The Gemma3n model to patch
        tokenizer: The tokenizer associated with the model
        original_text_vocab_size: Original text vocabulary size (auto-detected if None)
        new_text_vocab_size: New text vocabulary size (auto-detected if None)
        audio_vocab_size: Audio vocabulary size
        vision_vocab_size: Vision vocabulary size
        
    Returns:
        Patched model with updated configuration
    """
    
    # Auto-detect vocabulary sizes if not provided
    if original_text_vocab_size is None:
        # Try to get from model config
        if hasattr(model, 'config') and hasattr(model.config, 'vocab_size'):
            original_text_vocab_size = model.config.vocab_size
        else:
            # Fallback to common Gemma vocab size
            original_text_vocab_size = 256000
            logger.warning(f"Could not detect original vocab size, using default: {original_text_vocab_size}")
    
    if new_text_vocab_size is None:
        # Use current tokenizer size
        new_text_vocab_size = len(tokenizer)
        logger.info(f"Detected new text vocab size from tokenizer: {new_text_vocab_size}")
    
    # Log the changes
    logger.info(f"Applying vocab offset patch:")
    logger.info(f"  Original text vocab size: {original_text_vocab_size}")
    logger.info(f"  New text vocab size: {new_text_vocab_size}")
    logger.info(f"  Audio vocab size: {audio_vocab_size}")
    logger.info(f"  Vision vocab size: {vision_vocab_size}")
    
    # Apply the patch
    patched_model = apply_vocab_offset_patch(
        model=model,
        original_text_vocab_size=original_text_vocab_size,
        new_text_vocab_size=new_text_vocab_size,
        audio_vocab_size=audio_vocab_size,
        vision_vocab_size=vision_vocab_size
    )
    
    # Update model config if it exists
    if hasattr(patched_model, 'config'):
        vocab_config = VocabOffsetConfig(
            original_text_vocab_size=original_text_vocab_size,
            new_text_vocab_size=new_text_vocab_size,
            audio_vocab_size=audio_vocab_size,
            vision_vocab_size=vision_vocab_size
        )
        
        # Update config attributes
        patched_model.config.text_vocab_offset = vocab_config.text_vocab_offset
        patched_model.config.audio_vocab_offset = vocab_config.audio_vocab_offset
        patched_model.config.vision_vocab_offset = vocab_config.vision_vocab_offset
        patched_model.config.total_vocab_size = vocab_config.total_vocab_size
        
        logger.info("Updated model config with new vocab offsets")
    
    return patched_model


def safe_inference_with_patch(model, tokenizer, messages, max_new_tokens=128):
    """
    Safe inference function that handles vocab offset issues
    
    This function patches token ID ranges during inference to prevent out-of-range errors.
    """
    
    # Apply chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    
    # Check for potential out-of-range token IDs
    input_ids = inputs['input_ids']
    max_token_id = torch.max(input_ids).item()
    
    # Get model's expected vocab size
    model_vocab_size = getattr(model.config, 'vocab_size', len(tokenizer))
    
    if max_token_id >= model_vocab_size:
        logger.warning(f"Found token ID {max_token_id} >= model vocab size {model_vocab_size}")
        logger.warning("Clamping token IDs to valid range")
        inputs['input_ids'] = torch.clamp(input_ids, 0, model_vocab_size - 1)
    
    # Generate with safety checks
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        # Extract only newly generated tokens
        response_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        return response_text
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        # Try with a more conservative approach
        try:
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                max_new_tokens=min(max_new_tokens, 64),
                do_sample=False,  # Use greedy decoding as fallback
                pad_token_id=tokenizer.eos_token_id,
            )
            
            response_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            logger.warning("Used fallback generation strategy")
            return response_text
            
        except Exception as e2:
            logger.error(f"Fallback generation also failed: {e2}")
            return f"[Generation Error: {str(e2)}]"


def create_patched_inference_function(model, tokenizer):
    """
    Create a patched version of the inference function from CLAUDE.md
    
    This replaces the do_gemma_3n_inference function with safety patches.
    """
    
    def do_gemma_3n_inference_patched(messages, max_new_tokens=128):
        return safe_inference_with_patch(model, tokenizer, messages, max_new_tokens)
    
    return do_gemma_3n_inference_patched


def validate_model_compatibility(model, tokenizer):
    """
    Validate that the model and tokenizer are compatible with the patches
    """
    
    issues = []
    
    # Check tokenizer size
    tokenizer_size = len(tokenizer)
    if hasattr(model, 'config') and hasattr(model.config, 'vocab_size'):
        model_vocab_size = model.config.vocab_size
        if tokenizer_size != model_vocab_size:
            issues.append(f"Tokenizer size ({tokenizer_size}) != model vocab size ({model_vocab_size})")
    
    # Check for required attributes
    if not hasattr(model, 'generate'):
        issues.append("Model missing 'generate' method")
    
    # Check device compatibility
    if next(model.parameters()).device.type != 'cuda':
        issues.append("Model not on CUDA device (may cause performance issues)")
    
    if issues:
        logger.warning("Model compatibility issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False
    else:
        logger.info("Model compatibility validation passed")
        return True


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("Vocabulary Offset Patch Application Tool")
    print("=" * 50)
    print()
    print("This script applies vocab offset patches to fix token ID range issues.")
    print("Use this when you encounter out-of-range token ID errors during inference.")
    print()
    print("Example usage in your training script:")
    print()
    print("```python")
    print("from src.apply_vocab_patch import patch_gemma3n_model, create_patched_inference_function")
    print()
    print("# After loading your model and tokenizer")
    print("model, tokenizer = FastModel.from_pretrained(...)")
    print()
    print("# Apply the patch")
    print("model = patch_gemma3n_model(model, tokenizer)")
    print()
    print("# Create patched inference function")
    print("do_gemma_3n_inference = create_patched_inference_function(model, tokenizer)")
    print()
    print("# Use as normal")
    print("response = do_gemma_3n_inference(messages)")
    print("```")