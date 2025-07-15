"""
Gemma3n Core Module
モデル初期化、インファレンス、データ処理の核となる機能
"""

import os
import pandas as pd
import random
import numpy as np
import json
import time
import torch
from typing import Optional, List, Dict, Any, Tuple
from unsloth import FastModel
from transformers import TextStreamer
from torchvision import datasets, transforms, models

# モデルとトークナイザーのグローバル変数
model = None
tokenizer = None

def initialize_model():
    """Gemma3nモデルとトークナイザーを初期化"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        print("🚀 Gemma3nモデル初期化中...")
        model, tokenizer = FastModel.from_pretrained(
            model_name="unsloth/gemma-3n-E2B-it",
            dtype=None,
            max_seq_length=1024,
            load_in_4bit=True,
            full_finetuning=False,
        )
        print("✅ モデル初期化完了")
    
    return model, tokenizer

def do_gemma_3n_inference(messages, max_new_tokens=128):
    """Gemma3nでインファレンスを実行"""
    global model, tokenizer
    
    # モデルが初期化されていない場合は初期化
    if model is None or tokenizer is None:
        model, tokenizer = initialize_model()
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    
    # 常に出力をキャプチャして返す
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=1.0, top_p=0.95, top_k=64,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    # 新しく生成されたトークンのみを抽出
    response_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
    return response_text

def load_preprocessed_data():
    """前処理済みデータを高速読み込み"""
    try:
        if (os.path.exists('/kaggle/working/mushroom_dataset.parquet') and 
            os.path.exists('/kaggle/working/mushroom_metadata.json') and
            os.path.exists('/kaggle/working/mushroom_path_label.parquet')):
            
            print("📂 前処理済みデータを読み込み中...")
            
            # データセット読み込み
            data = pd.read_parquet('/kaggle/working/mushroom_dataset.parquet')
            
            # メタデータ読み込み
            with open('/kaggle/working/mushroom_metadata.json', 'r') as f:
                metadata = json.load(f)
            
            # path_label読み込み
            path_label_df = pd.read_parquet('/kaggle/working/mushroom_path_label.parquet')
            path_label = list(zip(path_label_df['path'], path_label_df['label']))
            
            print(f"✅ 前処理済みデータ読み込み完了:")
            print(f"   📊 Dataset: {len(data)} records")
            print(f"   🏷️ Classes: {metadata['total_classes']} classes")
            print(f"   🔗 Path-Label: {len(path_label)} records")
            
            return data, metadata, path_label
        else:
            return None, None, None
            
    except Exception as e:
        print(f"⚠️ 前処理済みデータ読み込み失敗: {str(e)}")
        return None, None, None

def process_dataset(dir0, max_files_per_class=1000):
    """データセットを前処理"""
    print("🔄 データセット前処理を実行中...")
    
    classes = []
    paths = []
    class_file_counts = {}
    
    for dirname, _, filenames in os.walk(dir0):
        class_name = dirname.split('/')[-1]
        if class_name not in class_file_counts:
            class_file_counts[class_name] = 0
        
        for filename in filenames:
            if class_file_counts[class_name] < max_files_per_class:
                classes += [class_name]
                paths += [(os.path.join(dirname, filename))]
                class_file_counts[class_name] += 1

    dataset0 = datasets.ImageFolder(root=dir0)
    class_names = dataset0.classes
    print(class_names)
    print(len(class_names))

    N = list(range(len(classes)))
    normal_mapping = dict(zip(class_names, N))
    reverse_mapping = dict(zip(N, class_names))

    data = pd.DataFrame(columns=['path', 'class', 'label'])
    data['path'] = paths
    data['class'] = classes
    data['label'] = data['class'].map(normal_mapping)
    print(len(data))

    def create_path_label_list(df):
        path_label_list = []
        for _, row in df.iterrows():
            path = row['path']
            label = row['label']
            path_label_list.append((path, label))
        return path_label_list

    path_label = create_path_label_list(data)
    path_label = random.sample(path_label, 20000)
    print(len(path_label))
    print(path_label[0:3])

    # データセット保存（効率化のため）
    print("\n💾 データセット保存中...")
    try:
        # Parquet形式で保存
        data.to_parquet('/kaggle/working/mushroom_dataset.parquet', index=False)
        
        # メタデータ保存
        metadata = {
            'class_names': class_names,
            'normal_mapping': normal_mapping,
            'reverse_mapping': reverse_mapping,
            'total_images': len(data),
            'total_classes': len(class_names),
            'path_label_sample_size': len(path_label)
        }
        
        with open('/kaggle/working/mushroom_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # path_labelも保存
        path_label_df = pd.DataFrame(path_label, columns=['path', 'label'])
        path_label_df.to_parquet('/kaggle/working/mushroom_path_label.parquet', index=False)
        
        print(f"✅ データセット保存完了:")
        print(f"   📊 Dataset: /kaggle/working/mushroom_dataset.parquet ({len(data)} records)")
        print(f"   🏷️ Metadata: /kaggle/working/mushroom_metadata.json")
        print(f"   🔗 Path-Label: /kaggle/working/mushroom_path_label.parquet ({len(path_label)} records)")
        
    except Exception as e:
        print(f"⚠️ データセット保存失敗: {str(e)}")
    
    return data, class_names, normal_mapping, reverse_mapping, path_label

def get_model_tokenizer():
    """モデルとトークナイザーを取得（外部アクセス用）"""
    global model, tokenizer
    if model is None or tokenizer is None:
        return initialize_model()
    return model, tokenizer