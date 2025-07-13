#%%
# Vision Fine-tuning for Mushroom Classification - CLAUDE.md Based
import os
import random
import pandas as pd
from typing import List
from torchvision import datasets
from unsloth import FastModel, is_bf16_supported, FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

#%%
# Load the model (exactly as in CLAUDE.md)
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E2B-it",
    dtype = None,
    max_seq_length = 1024,
    load_in_4bit = True,
    full_finetuning = False,
)

#%%
# Data loading (simplified)
dir0 = '/kaggle/input/mushroom1/merged_dataset'
max_files_per_class = 50  # Smaller for testing

print("🔄 Loading mushroom dataset...")
classes = []
paths = []
class_file_counts = {}

for dirname, _, filenames in os.walk(dir0):
    class_name = dirname.split('/')[-1]
    if class_name not in class_file_counts:
        class_file_counts[class_name] = 0
    
    for filename in filenames:
        if class_file_counts[class_name] < max_files_per_class:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                full_path = os.path.join(dirname, filename)
                if os.path.exists(full_path):
                    classes.append(class_name)
                    paths.append(full_path)
                    class_file_counts[class_name] += 1

dataset0 = datasets.ImageFolder(root=dir0)
class_names = dataset0.classes
print(f"✅ Loaded {len(paths)} images from {len(class_names)} classes")

# Create simple dataset list
dataset = []
for i, (path, class_name) in enumerate(zip(paths, classes)):
    dataset.append({"path": path, "class": class_name})

# Sample for manageable training
sample_size = min(100, len(dataset))  # Very small for testing
dataset = random.sample(dataset, sample_size)
print(f"📊 Using {len(dataset)} samples for training")

#%%
# Convert to conversation format (exactly as in CLAUDE.md)
instruction = "You are an expert of mushroom. Describe accurately what you see in this image."

def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "image", "image" : sample["path"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["class"]} ]
        },
    ]
    return { "messages" : conversation }

converted_dataset = [convert_to_conversation(sample) for sample in dataset]
print(f"✅ Converted {len(converted_dataset)} samples to conversation format")

# Show sample
print("\n📋 Sample conversation:")
print(converted_dataset[0])

#%%
# Vision Fine-tuning (exactly as in CLAUDE.md)
print("\n🚀 Starting Vision Fine-tuning...")

# Add PEFT configuration for quantized model
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers = True,
    finetune_language_layers = False,
    r = 16,  # LoRA rank
    lora_alpha = 32,
    lora_dropout = 0.1,
    bias = "none",
    random_state = 3407,
    use_gradient_checkpointing = "unsloth",
)

# Enable for training
FastVisionModel.for_training(model)

# Create trainer (exactly as in CLAUDE.md)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 1,  # Reduced for memory
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 10,  # Very short for testing
        learning_rate = 2e-4,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "./mushroom_vision_outputs",
        report_to = "none",

        # You MUST put the below items for vision finetuning (from CLAUDE.md):
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 1,
        max_seq_length = 2048,
    ),
)

# Start training
print("📈 Training started...")
trainer.train()

print("🎉 Training completed!")
print(f"💾 Model saved to: ./mushroom_vision_outputs")

#%%
# Test the model
print("\n🧪 Testing model...")

# Get a test sample
test_sample = dataset[0]
test_messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "What type of mushroom is this?"},
        {"type": "image", "image": test_sample["path"]}
    ]
}]

# Generate response
inputs = tokenizer.apply_chat_template(
    test_messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

response_tokens = outputs[0][inputs['input_ids'].shape[1]:]
response = tokenizer.decode(response_tokens, skip_special_tokens=True)

print(f"✅ Test completed!")
print(f"   Test image: {os.path.basename(test_sample['path'])}")
print(f"   Actual class: {test_sample['class']}")
print(f"   Model response: {response}")