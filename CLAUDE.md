

# 推論の基本構成
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
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        tokenize = True,
        return_dict = True,
        return_tensors = "pt",
    ).to("cuda")
    
    # 常に出力をキャプチャして返す
    outputs = model.generate(
        **inputs,
        max_new_tokens = max_new_tokens,
        temperature = 1.0, top_p = 0.95, top_k = 64,
        do_sample = True,
        pad_token_id = tokenizer.eos_token_id,
    )
    # 新しく生成されたトークンのみを抽出
    response_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
    return response_text

# データセット読み込み
classes=[]
paths=[]
class_file_counts = {}
for dirname, _, filenames in os.walk(dir0):
    class_name = dirname.split('/')[-1]
    if class_name not in class_file_counts:
        class_file_counts[class_name] = 0
    
    for filename in filenames:
        if class_file_counts[class_name] < max_files_per_class:
            classes+=[class_name]
            paths += [(os.path.join(dirname, filename))]
            class_file_counts[class_name] += 1

dataset0=datasets.ImageFolder(root=dir0)
class_names=dataset0.classes
print(class_names)
print(len(class_names))

N=list(range(len(classes)))
normal_mapping=dict(zip(class_names,N))
reverse_mapping=dict(zip(N,class_names))

data=pd.DataFrame(columns=['path','class','label'])
data['path']=paths
data['class']=classes
data['label']=data['class'].map(normal_mapping)
print(len(data))

def create_path_label_list(df):
    path_label_list = []
    for _, row in df.iterrows():
        path = row['path']
        label = row['label']
        path_label_list.append((path, label))
    return path_label_list

path_label = create_path_label_list(data)
path_label = random.sample(path_label,20000)

# conversation作成
instruction = "You are an expert radiographer. Describe accurately what you see in this image."

def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "image", "image" : sample["path"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["label"]} ]
        },
    ]
    return { "messages" : conversation }
pass

converted_dataset = [convert_to_conversation(sample) for sample in dataset]

converted_dataset[0]

{'messages': [{'role': 'user',
   'content': [{'type': 'text',
     'text': 'You are an expert radiographer. Describe accurately what you see in this image.'},
    {'type': 'image',
     'image': <PIL.PngImagePlugin.PngImageFile image mode=L size=657x442>}]},
  {'role': 'assistant',
   'content': [{'type': 'text',
     'text': 'Panoramic radiography shows an osteolytic lesion in the right posterior maxilla with resorption of the floor of the maxillary sinus (arrows).'}]}]}


# SFT training
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 30,
        # num_train_epochs = 1, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",     # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = 2048,
    ),
)