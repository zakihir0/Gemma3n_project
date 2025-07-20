# Vocabulary Offset Patch for Gemma3n Models

This patch addresses vocabulary offset issues when extending text vocabulary in multimodal Gemma3n models. It implements the following fixes:

## Problem Solved

When extending the text vocabulary of Gemma3n models, the original `config.audio_vocab_offset` and `config.vision_vocab_offset` become incorrect, leading to:
- Out-of-range token ID errors during inference
- Incorrect token type classification  
- Model compatibility issues with extended vocabularies

## Solution Implemented

### 1. Vocabulary Offset Recalculation (`src/vocab_offset_patch.py`)

- **VocabOffsetConfig**: Automatically calculates correct offsets:
  ```
  text_vocab_offset = 0
  audio_vocab_offset = new_text_vocab_size  
  vision_vocab_offset = new_text_vocab_size + audio_vocab_size
  ```

- **PatchedEmbedding**: Handles token ID ranges with strict validation:
  ```python
  # Text tokens: 0 to new_text_vocab_size-1
  # Audio tokens: audio_vocab_offset to vision_vocab_offset-1  
  # Vision tokens: vision_vocab_offset to total_vocab_size-1
  ```

### 2. Configuration Updates (`src/config_updater.py`)

- Updates model configuration files with correct vocab offsets
- Validates offset ordering and consistency
- Maintains backward compatibility with existing configs

### 3. Model Patching (`src/apply_vocab_patch.py`)

- **patch_gemma3n_model()**: Applies all patches to loaded models
- **safe_inference_with_patch()**: Handles out-of-range token IDs during generation
- **create_patched_inference_function()**: Drop-in replacement for existing inference

### 4. Forward Pass Judgment Condition Fix

The original condition `if token_id >= offset` is replaced with strict range checking:
```python
# OLD (incorrect)
if token_id >= audio_vocab_offset:
    # Handle as audio/vision

# NEW (correct) 
if audio_vocab_offset <= token_id < vision_vocab_offset:
    # Handle as audio
elif vision_vocab_offset <= token_id < total_vocab_size:
    # Handle as vision
```

## Integration Guide

### Quick Integration

Replace your existing model loading with:

```python
# Import the patch
from src.apply_vocab_patch import patch_gemma3n_model, create_patched_inference_function

# Load model as usual
model, tokenizer = FastVisionModel.from_pretrained(...)

# Apply patches  
model = patch_gemma3n_model(model, tokenizer)

# Create safe inference function
do_gemma_3n_inference = create_patched_inference_function(model, tokenizer)

# Use normally - the inference function handles all edge cases
response = do_gemma_3n_inference(messages, max_new_tokens=128)
```

### Advanced Usage

For custom vocab sizes or config updates:

```python
from src.vocab_offset_patch import VocabOffsetConfig, VocabOffsetPatcher
from src.config_updater import ConfigUpdater

# Custom configuration
vocab_config = VocabOffsetConfig(
    original_text_vocab_size=256000,
    new_text_vocab_size=260000,
    audio_vocab_size=2048,
    vision_vocab_size=2048
)

# Apply patches manually
patcher = VocabOffsetPatcher(vocab_config)
model = patcher.patch_model(model)

# Update config files
updater = ConfigUpdater("config.json")
updater.update_vocab_offsets(256000, 260000, 2048, 2048)
updater.save_config("updated_config.json")
```

## Files Created

- **`src/vocab_offset_patch.py`**: Core patching logic and offset calculations
- **`src/config_updater.py`**: Configuration file update utilities  
- **`src/apply_vocab_patch.py`**: High-level integration functions
- **`vocab_patch_example.py`**: Complete usage example

## Features

✅ **Automatic Offset Calculation**: Correctly shifts audio/vision offsets after text vocab expansion

✅ **Strict Token ID Validation**: Prevents out-of-range errors with proper range checking

✅ **Backward Compatibility**: Works with existing training code and model weights

✅ **Safe Inference**: Handles edge cases with fallback generation strategies

✅ **Configuration Management**: Updates and validates model config files

✅ **Drop-in Replacement**: Minimal changes required to existing code

## Testing

Run the example to test the implementation:

```bash
python vocab_patch_example.py
```

This will demonstrate:
- Model loading with patches applied
- Safe inference with out-of-range token handling
- Token type classification validation
- Configuration updates

## Implementation Notes

### Risk Mitigation

- **Existing Weights Compatibility**: Preserves original text token embeddings
- **Gradual Fallback**: Multiple generation strategies if errors occur
- **Validation**: Extensive config and model compatibility checking
- **Logging**: Detailed logging for debugging and monitoring

### Performance Considerations

- **Minimal Overhead**: Patches only add lightweight validation logic
- **Memory Efficient**: Reuses existing embeddings where possible
- **CUDA Compatible**: Maintains GPU acceleration and device placement

### Limitations

- Requires the underlying model to support the expected embedding structure
- Audio/vision embedding modules must be present for full functionality
- Some advanced multimodal features may need additional patches

## Troubleshooting

### Common Issues

**Out-of-range token ID errors**: 
- Ensure `patch_gemma3n_model()` is called after loading
- Check tokenizer size matches expected new vocab size

**Config validation failures**:
- Verify offset ordering (text < audio < vision)
- Check total vocab size calculations

**Generation failures**:
- Use `safe_inference_with_patch()` for robust error handling
- Enable logging to see detailed error information

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

This provides verbose output about:
- Offset calculations and updates
- Token ID validation results  
- Generation strategy selection
- Config validation status