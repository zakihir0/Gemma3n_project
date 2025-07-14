#%%
# Vision Fine-tuning for Mushroom Classification - CLAUDE.md Based

!pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
!pip install pip3-autoremove
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu124
!pip install unsloth
!pip install faiss-cpu
!pip install sentence-transformers
!pip install wikipedia
!pip install --no-deps git+https://github.com/huggingface/transformers.git
!pip install --no-deps --upgrade timm

#%%
import os
import random
import pandas as pd
import json
from typing import List
from torchvision import datasets
from unsloth import FastModel, is_bf16_supported, FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

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

json_path = f"/kaggle/working/mushroom_dataset.json"

with open(json_path, 'w')as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

#%%
json_path = f"/kaggle/working/mushroom_dataset.json"
with open(json_path, 'r') as f:
    dataset = json.load(f)

# Sample for manageable training
sample_size = min(len(dataset), len(dataset))  # Very small for testing
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
# Split dataset into train and validation
from sklearn.model_selection import train_test_split

# Extract class labels for stratified split
class_labels = [sample['messages'][1]['content'][0]['text'] for sample in converted_dataset]

# Split with stratification to maintain class balance
train_dataset, val_dataset = train_test_split(
    converted_dataset, 
    test_size=0.2, 
    random_state=3407,
    stratify=class_labels
)

print(f"📊 Dataset split:")
print(f"   Training samples: {len(train_dataset)}")
print(f"   Validation samples: {len(val_dataset)}")

#%%
# Vision Fine-tuning (exactly as in CLAUDE.md)
print("\n🚀 Starting Vision Fine-tuning...")

# Add PEFT configuration for quantized model
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers = True,
    finetune_language_layers = False,
    finetune_attention_modules = True,
    finetune_mlp_modules = True,
    r = 16,  # LoRA rank
    lora_alpha = 32,
    lora_dropout = 0,  # Changed from 0.1 to 0 for Unsloth compatibility
    bias = "none",
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
    random_state = 3407,
    use_gradient_checkpointing = "unsloth",
)

# Enable for training
FastVisionModel.for_training(model)

# Create trainer with validation dataset and evaluation settings
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = train_dataset,  # Use split training data
    eval_dataset = val_dataset,     # Add validation data
    args = SFTConfig(
        per_device_train_batch_size = 5,  # Reduced for memory
        per_device_eval_batch_size = 5,   # Validation batch size
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 100,  # Very short for testing
        learning_rate = 1e-4,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "./mushroom_vision_outputs",
        report_to = "none",
        
        # Evaluation settings
        eval_strategy = "steps",         # Evaluate every eval_steps (corrected parameter name)
        eval_steps = 10,                 # Evaluate every 10 steps
        save_strategy = "steps",         # Save every save_steps
        save_steps = 10,                 # Save every 10 steps
        load_best_model_at_end = True,   # Load best model at end
        metric_for_best_model = "eval_loss",  # Use eval_loss for best model
        greater_is_better = False,       # Lower eval_loss is better

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
# %%
import torch
import gc

# del model
# del trainer
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
gc.collect()

# %%
