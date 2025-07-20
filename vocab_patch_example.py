"""
Example: Integrating Vocabulary Offset Patches with Gemma3n Training

This example shows how to apply the vocab offset patches to your existing training pipeline
to fix the config.audio_vocab_offset / vision_vocab_offset issues.
"""

import os
import torch
import logging
from unsloth import FastVisionModel

# Import our patch modules
from src.apply_vocab_patch import patch_gemma3n_model, create_patched_inference_function, validate_model_compatibility
from src.vocab_offset_patch import VocabOffsetConfig
from src.config_updater import create_default_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Example of applying vocab offset patches to Gemma3n model
    """
    
    print("Loading Gemma3n model with vocab offset patches...")
    print("=" * 60)
    
    # 1. Load the model as usual (from CLAUDE.md)
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name="unsloth/gemma-3n-E2B-it",
        dtype=None,
        max_seq_length=1024,
        load_in_4bit=True,
        full_finetuning=False,
    )
    
    print(f"Original model vocab size: {getattr(model.config, 'vocab_size', 'Unknown')}")
    print(f"Tokenizer size: {len(tokenizer)}")
    
    # 2. Validate model compatibility
    if not validate_model_compatibility(model, tokenizer):
        print("Warning: Model compatibility issues detected")
    
    # 3. Apply vocab offset patches
    print("\nApplying vocabulary offset patches...")
    
    # You can specify custom vocab sizes or let them be auto-detected
    patched_model = patch_gemma3n_model(
        model=model,
        tokenizer=tokenizer,
        original_text_vocab_size=256000,  # Gemma's original vocab size
        new_text_vocab_size=None,         # Auto-detect from tokenizer
        audio_vocab_size=1024,            # Standard audio vocab size
        vision_vocab_size=1024            # Standard vision vocab size
    )
    
    # 4. Create patched inference function (replaces do_gemma_3n_inference from CLAUDE.md)
    do_gemma_3n_inference = create_patched_inference_function(patched_model, tokenizer)
    
    print("\nPatch applied successfully!")
    
    # 5. Test the patched inference
    print("\nTesting patched inference...")
    
    # Example messages (from CLAUDE.md style)
    test_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "You are an expert radiographer. Describe what you see."},
                # Note: In real usage, you'd include an actual image here
                {"type": "text", "text": "[Image would be here]"}
            ]
        }
    ]
    
    try:
        response = do_gemma_3n_inference(test_messages, max_new_tokens=50)
        print(f"Inference successful: {response[:100]}...")
        
    except Exception as e:
        print(f"Inference failed: {e}")
        
    # 6. Show the updated configuration
    print("\nUpdated vocabulary configuration:")
    if hasattr(patched_model, 'config'):
        config = patched_model.config
        print(f"  Text vocab offset: {getattr(config, 'text_vocab_offset', 'Not set')}")
        print(f"  Audio vocab offset: {getattr(config, 'audio_vocab_offset', 'Not set')}")
        print(f"  Vision vocab offset: {getattr(config, 'vision_vocab_offset', 'Not set')}")
        print(f"  Total vocab size: {getattr(config, 'total_vocab_size', 'Not set')}")
    
    # 7. Demonstrate strict token ID validation
    print("\nDemonstrating token ID validation...")
    
    # Create some test token IDs that might be out of range
    test_input_ids = torch.tensor([[0, 100, 50000, 260000, 300000]])  # Some potentially out-of-range IDs
    
    vocab_config = VocabOffsetConfig(
        original_text_vocab_size=256000,
        new_text_vocab_size=len(tokenizer),
        audio_vocab_size=1024,
        vision_vocab_size=1024
    )
    
    from src.vocab_offset_patch import VocabOffsetPatcher
    patcher = VocabOffsetPatcher(vocab_config)
    token_masks = patcher.create_token_type_mask(test_input_ids)
    
    print("Token type classification:")
    for i, token_id in enumerate(test_input_ids[0]):
        token_id_val = token_id.item()
        print(f"  Token ID {token_id_val}: ", end="")
        
        if token_masks["text"][0][i]:
            print("TEXT")
        elif token_masks["audio"][0][i]:
            print("AUDIO")
        elif token_masks["vision"][0][i]:
            print("VISION")
        else:
            print("OUT OF RANGE")
    
    print("\nExample complete!")
    print("\nTo integrate this into your training script, add these lines after loading your model:")
    print("```python")
    print("from src.apply_vocab_patch import patch_gemma3n_model, create_patched_inference_function")
    print("model = patch_gemma3n_model(model, tokenizer)")
    print("do_gemma_3n_inference = create_patched_inference_function(model, tokenizer)")
    print("```")


def integration_example_with_existing_code():
    """
    Example showing how to modify the existing training code from CLAUDE.md
    """
    
    print("\n" + "=" * 60)
    print("INTEGRATION EXAMPLE")
    print("=" * 60)
    
    # This is how you would modify the existing code from CLAUDE.md:
    
    print("BEFORE (from CLAUDE.md):")
    print("-" * 30)
    print("""
# Load the model
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E2B-it",
    dtype = None,
    max_seq_length = 1024,
    load_in_4bit = True,
    full_finetuning = False,
)

# Helper function for inference
def do_gemma_3n_inference(messages, max_new_tokens = 128):
    inputs = tokenizer.apply_chat_template(...)
    outputs = model.generate(...)
    return response_text
""")
    
    print("\nAFTER (with patches):")
    print("-" * 30)
    print("""
# Import patches
from src.apply_vocab_patch import patch_gemma3n_model, create_patched_inference_function

# Load the model
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E2B-it",
    dtype = None,
    max_seq_length = 1024,
    load_in_4bit = True,
    full_finetuning = False,
)

# Apply vocab offset patches
model = patch_gemma3n_model(model, tokenizer)

# Create patched inference function (replaces the manual implementation)
do_gemma_3n_inference = create_patched_inference_function(model, tokenizer)
""")
    
    print("\nThe rest of your training code remains the same!")
    print("The patched inference function will handle:")
    print("  ✓ Out-of-range token ID detection and clamping")
    print("  ✓ Proper vocab offset calculations")
    print("  ✓ Safe generation with fallback strategies")
    print("  ✓ Compatibility with existing chat template format")


if __name__ == "__main__":
    try:
        main()
        integration_example_with_existing_code()
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure the src/ directory is in your Python path")
        print("You may need to run: export PYTHONPATH=$PYTHONPATH:.")
        
    except Exception as e:
        print(f"Error running example: {e}")
        logger.exception("Full error details:")