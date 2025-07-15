#%%
# Vision Fine-tuning for Mushroom Classification - CLAUDE.md Based

!pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
!pip install --upgrade bitsandbytes
!pip install triton==3.2.0
!pip install pip3-autoremove
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu124
!pip install unsloth
!pip install faiss-cpu
!pip install sentence-transformers
!pip install wikipedia
!pip install --no-deps git+https://github.com/huggingface/transformers.git
!pip install --no-deps --upgrade timm
!pip install wandb
!pip uninstall deepspeed -y


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
import wandb
from PIL import Image
from IPython.display import display, Image as IPImage

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
os.environ["WANDB_API_KEY"] = "abd0ce2837eca439d66f6c603136603c1729cd3e"
os.environ["DEEPSPEED_DISABLE"] = "1"  # Disable DeepSpeed to avoid torch.distributed.elastic conflicts
os.environ["ACCELERATE_DISABLE_RICH"] = "1"  # Disable rich formatting to avoid conflicts
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism

#%%
# Initialize WandB with environment variable login
wandb.login(key=os.getenv("WANDB_API_KEY"))

wandb.init(
    project="gemma3n-mushroom-classification",
    name="vision-finetuning-experiment",
    config={
        "model_name": "unsloth/gemma-3n-E2B-it",
        "max_seq_length": 1024,
        "load_in_4bit": True,
        "max_files_per_class": 50,
        "lora_rank": 16,
        "lora_alpha": 32,
    }
)

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
dir0 = '/notebooks/kaggle/input/mushroom1/merged_dataset'
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

# Create simple dataset list
dataset = []
for i, (path, class_name) in enumerate(zip(paths, classes)):
    dataset.append({"path": path, "class": class_name})

json_path = f"/notebooks/kaggle/working/mushroom_dataset.json"

with open(json_path, 'w')as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

#%%
json_path = f"/notebooks/kaggle/working/mushroom_dataset.json"
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
    try:
        # Load image as PIL Image object instead of path string
        image = Image.open(sample["path"])
        # Convert to RGB if needed (handles RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        conversation = [
            { "role": "user",
              "content" : [
                {"type" : "text",  "text"  : instruction},
                {"type" : "image", "image" : image} ]
            },
            { "role" : "assistant",
              "content" : [
                {"type" : "text",  "text"  : sample["class"]} ]
            },
        ]
        return { "messages" : conversation }
    except Exception as e:
        print(f"❌ Error loading image {sample['path']}: {e}")
        return None

converted_dataset = [convert_to_conversation(sample) for sample in dataset]
# Filter out None values (failed image loads)
converted_dataset = [item for item in converted_dataset if item is not None]
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
    r = 32,  # LoRA rank
    lora_alpha = 64,
    lora_dropout = 0,  # Changed from 0.1 to 0 for Unsloth compatibility
    bias = "none",
    # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj","vision_embed","tok_embeddings"],
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
    # eval_dataset = val_dataset,     # Remove validation data for simplicity
    args = SFTConfig(
        per_device_train_batch_size = 8,  # Further reduced for stability
        per_device_eval_batch_size = 2,   # Validation batch size
        gradient_accumulation_steps = 8,  # Increased to maintain effective batch size
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
        report_to = "wandb",
        run_name = "gemma3n-mushroom-vision-ft",
        deepspeed = None,
        dataloader_pin_memory = False,  # Disable pin memory to avoid conflicts
        dataloader_num_workers = 0,    # Disable multiprocessing to avoid issues
        
        # Simplified evaluation settings
        eval_strategy = "no",            # Disable evaluation to simplify setup
        save_strategy = "steps",         # Save every save_steps
        save_steps = 50,                 # Less frequent saves

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

# Log training completion to WandB
wandb.log({"training_status": "completed"})
wandb.finish()

#%%
# Test the model
print("\n🧪 Testing model...")

# Get a test sample and load as PIL Image
test_sample = dataset[0]
test_image = Image.open(test_sample["path"])
if test_image.mode != 'RGB':
    test_image = test_image.convert('RGB')

test_messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "What type of mushroom is this?"},
        {"type": "image", "image": test_image}
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
print(f"\n🖼️ Test Image: {os.path.basename(test_sample['path'])}")
print("=" * 50)

# Display the test image
display(test_image)

print(f"🏷️ Actual class: {test_sample['class']}")
print(f"🤖 Model response: {response}")

# Extract predicted class name from response
def extract_predicted_class(response_text, available_classes):
    """Extract the most likely class name from model response"""
    response_lower = response_text.lower()
    
    # Look for exact matches first
    for class_name in available_classes:
        if class_name.lower() in response_lower:
            return class_name
    
    # If no exact match, look for partial matches
    for class_name in available_classes:
        class_words = class_name.lower().split()
        if any(word in response_lower for word in class_words if len(word) > 3):
            return class_name
    
    return None

class_names = dataset0.classes
predicted_class = extract_predicted_class(response, class_names)

if predicted_class:
    print(f"🔍 Extracted predicted class: {predicted_class}")
    
    # Find examples of the predicted class
    predicted_class_samples = [item for item in dataset if item['class'] == predicted_class]
    
    if predicted_class_samples:
        print(f"\n📚 Reference images for '{predicted_class}' (showing 3 examples):")
        print("-" * 50)
        
        # Show up to 3 reference images
        reference_samples = random.sample(predicted_class_samples, min(3, len(predicted_class_samples)))
        
        for i, ref_sample in enumerate(reference_samples, 1):
            ref_image = Image.open(ref_sample["path"])
            if ref_image.mode != 'RGB':
                ref_image = ref_image.convert('RGB')
            
            print(f"Reference {i}: {os.path.basename(ref_sample['path'])}")
            display(ref_image)
            print()
    else:
        print(f"❌ No reference images found for '{predicted_class}'")
else:
    print("❓ Could not extract a clear class prediction from the response")

# Check if prediction is correct
is_correct = test_sample['class'].lower() in response.lower()
result_emoji = "✅" if is_correct else "❌"
print(f"{result_emoji} Prediction: {'Correct' if is_correct else 'Incorrect'}")

# Log to WandB
wandb_log_data = {
    "single_test/test_image": wandb.Image(test_image, caption=f"Test: {test_sample['class']}"),
    "single_test/actual_class": test_sample['class'],
    "single_test/predicted_response": response,
    "single_test/is_correct": is_correct
}

if predicted_class:
    wandb_log_data["single_test/extracted_class"] = predicted_class
    
    # Log reference images if available
    if predicted_class_samples:
        for j, ref_sample in enumerate(reference_samples[:2], 1):  # Log first 2 reference images
            ref_img = Image.open(ref_sample["path"])
            if ref_img.mode != 'RGB':
                ref_img = ref_img.convert('RGB')
            wandb_log_data[f"single_test/reference_{j}"] = wandb.Image(
                ref_img, 
                caption=f"Ref {j}: {predicted_class}"
            )

wandb.log(wandb_log_data)

#%%
# Multiple test cases from different classes
print("\n🧪 Testing multiple samples from different classes...")
print("=" * 60)

# Get unique classes and select one sample from each
unique_classes = list(set([item['class'] for item in dataset]))
test_samples = []

for class_name in unique_classes[:5]:  # Test first 5 classes
    class_samples = [item for item in dataset if item['class'] == class_name]
    if class_samples:
        test_samples.append(random.choice(class_samples))

# Create WandB Table for multiple test results
table_columns = ["Test Image", "Actual Class", "Predicted Response", "Extracted Class", "Is Correct", "Reference 1", "Reference 2"]
test_table = wandb.Table(columns=table_columns)
correct_count = 0

for i, sample in enumerate(test_samples, 1):
    print(f"\n📸 Test Case {i}/5: {sample['class']}")
    print("-" * 40)
    
    # Load and display image
    test_img = Image.open(sample["path"])
    if test_img.mode != 'RGB':
        test_img = test_img.convert('RGB')
    
    display(test_img)
    
    # Generate prediction
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What type of mushroom is this?"},
            {"type": "image", "image": test_img}
        ]
    }]
    
    inputs = tokenizer.apply_chat_template(
        messages,
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
    prediction = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    # Display results
    print(f"🏷️ Actual: {sample['class']}")
    print(f"🤖 Predicted: {prediction}")
    
    # Extract and show reference images for predicted class
    predicted_class = extract_predicted_class(prediction, class_names)
    
    if predicted_class:
        print(f"🔍 Extracted class: {predicted_class}")
        predicted_class_samples = [item for item in dataset if item['class'] == predicted_class]
        
        if predicted_class_samples and len(predicted_class_samples) > 0:
            print(f"📚 Reference examples for '{predicted_class}':")
            reference_samples = random.sample(predicted_class_samples, min(2, len(predicted_class_samples)))
            
            for j, ref_sample in enumerate(reference_samples, 1):
                ref_image = Image.open(ref_sample["path"])
                if ref_image.mode != 'RGB':
                    ref_image = ref_image.convert('RGB')
                
                print(f"  Ref {j}: {os.path.basename(ref_sample['path'])}")
                display(ref_image)
    
    is_match = sample['class'].lower() in prediction.lower()
    status_emoji = "✅" if is_match else "❌"
    print(f"{status_emoji} {'CORRECT' if is_match else 'INCORRECT'}")
    
    if is_match:
        correct_count += 1
    
    # Add row to WandB table
    table_row = [
        wandb.Image(test_img, caption=f"Test: {sample['class']}"),
        sample['class'],
        prediction,
        predicted_class if predicted_class else "N/A",
        is_match
    ]
    
    # Add reference images to table
    if predicted_class and predicted_class_samples:
        for k in range(2):  # Add up to 2 reference images
            if k < len(reference_samples):
                ref_img = Image.open(reference_samples[k]["path"])
                if ref_img.mode != 'RGB':
                    ref_img = ref_img.convert('RGB')
                table_row.append(wandb.Image(ref_img, caption=f"Ref: {predicted_class}"))
            else:
                table_row.append(None)  # Empty cell if no reference image
    else:
        table_row.extend([None, None])  # Empty cells for reference images
    
    test_table.add_data(*table_row)
    
    if i < len(test_samples):
        print("\n" + "="*60)

# Summary statistics
accuracy = (correct_count / len(test_samples)) * 100 if test_samples else 0

print(f"\n📊 Test Summary:")
print("=" * 30)
print(f"🧬 Classes tested: {len(test_samples)}")
print(f"🔬 Total available classes: {len(unique_classes)}")
print(f"🎯 Accuracy: {accuracy:.1f}% ({correct_count}/{len(test_samples)})")

# Log comprehensive results to WandB
wandb.log({
    "multiple_tests/test_results_table": test_table,
    "multiple_tests/accuracy": accuracy,
    "multiple_tests/correct_predictions": correct_count,
    "multiple_tests/total_tests": len(test_samples),
    "multiple_tests/total_classes": len(unique_classes)
})
# %%
import torch
import gc

# del model
# del trainer
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
gc.collect()

# %%
