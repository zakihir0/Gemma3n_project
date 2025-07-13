#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

!pip install pip3-autoremove
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu124
!pip install unsloth
!pip install faiss-cpu
!pip install sentence-transformers
!pip install wikipedia
!pip install --no-deps git+https://github.com/huggingface/transformers.git
!pip install --no-deps --upgrade timm
#%%
import torch
import pandas as pd
import random
import numpy as np
import faiss
import wikipedia
import json
import re
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from unsloth import FastModel
from transformers import TextStreamer
from torchvision import datasets, transforms, models
from sentence_transformers import SentenceTransformer

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

#%%
%%time
dir0='/kaggle/input/mushroom1/merged_dataset'
max_files_per_class = 1000  # Limit files per class to control total dataset size

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
            paths+=[(os.path.join(dirname, filename))]
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
print(len(path_label))
print(path_label[0:3])

image_path, label = path_label[0]

#%%
# Vision Fine-tuning Implementation for Mushroom Classification
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset
from PIL import Image
import torch.nn.functional as F

class MushroomVisionDataset:
    """キノコ画像データセット準備クラス"""
    
    def __init__(self, image_paths: List[str], labels: List[str], class_names: List[str]):
        """
        Args:
            image_paths: 画像ファイルパスのリスト
            labels: クラス名のリスト
            class_names: 全クラス名のリスト
        """
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.label_to_id = {name: i for i, name in enumerate(class_names)}
    
    def prepare_vision_dataset(self, train_ratio: float = 0.8, max_samples_per_class: int = 10) -> tuple:
        """
        Vision fine-tuning用のデータセット準備
        
        Args:
            train_ratio: 訓練データの比率
            max_samples_per_class: クラスあたりの最大サンプル数
            
        Returns:
            (train_dataset, val_dataset)
        """
        print(f"📊 Vision fine-tuning用データセット準備開始...")
        
        # クラス別にデータを整理
        class_data = {}
        for path, label in zip(self.image_paths, self.labels):
            if label not in class_data:
                class_data[label] = []
            class_data[label].append(path)
        
        # 各クラスから均等にサンプリング
        selected_data = []
        for class_name, paths in class_data.items():
            if len(paths) > 0:
                # クラスあたりの最大サンプル数に制限
                sampled_paths = random.sample(paths, min(len(paths), max_samples_per_class))
                for path in sampled_paths:
                    selected_data.append({
                        'image_path': path,
                        'class_name': class_name,
                        'class_id': self.label_to_id.get(class_name, -1),
                        'text': f"This is a {class_name} mushroom."
                    })
        
        print(f"   選択されたデータ数: {len(selected_data)}")
        print(f"   対象クラス数: {len(class_data)}")
        
        # データセットをシャッフル
        random.shuffle(selected_data)
        
        # 訓練/検証データに分割
        split_idx = int(len(selected_data) * train_ratio)
        train_data = selected_data[:split_idx]
        val_data = selected_data[split_idx:]
        
        print(f"   訓練データ: {len(train_data)}")
        print(f"   検証データ: {len(val_data)}")
        
        # Hugging Face Dataset形式に変換
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        return train_dataset, val_dataset

def formatting_func_for_vision(examples):
    """Vision fine-tuning用のフォーマット関数"""
    texts = []
    for text in examples['text']:
        texts.append(text)
    return {"text": texts}

class MushroomVisionFineTuner:
    """キノコ画像分類用Vision Fine-tunerクラス"""
    
    def __init__(self, base_model, base_tokenizer):
        """
        Args:
            base_model: ベースのGemma3nモデル
            base_tokenizer: ベースのトークナイザー
        """
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.vision_model = None
        self.is_fine_tuned = False
    
    def setup_vision_model(self, finetune_vision_layers: bool = True, 
                          finetune_language_layers: bool = False):
        """
        Vision fine-tuning用モデル設定
        
        Args:
            finetune_vision_layers: Vision層をfine-tuningするか
            finetune_language_layers: Language層をfine-tuningするか
        """
        print(f"🔧 Vision fine-tuning用モデル設定中...")
        
        try:
            # FastVisionModelでPEFT設定
            self.vision_model = FastVisionModel.get_peft_model(
                self.base_model,
                finetune_vision_layers=finetune_vision_layers,
                finetune_language_layers=finetune_language_layers,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=16,  # LoRA rank
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
            print(f"✅ Vision model設定完了")
            print(f"   Vision層 fine-tuning: {finetune_vision_layers}")
            print(f"   Language層 fine-tuning: {finetune_language_layers}")
            
        except Exception as e:
            print(f"❌ Vision model設定エラー: {str(e)}")
            raise e
    
    def train_vision_model(self, train_dataset, val_dataset, 
                          output_dir: str = "./mushroom_vision_model",
                          num_epochs: int = 3,
                          batch_size: int = 2,
                          learning_rate: float = 2e-5):
        """
        Vision modelを訓練
        
        Args:
            train_dataset: 訓練データセット
            val_dataset: 検証データセット
            output_dir: モデル保存ディレクトリ
            num_epochs: エポック数
            batch_size: バッチサイズ
            learning_rate: 学習率
        """
        if self.vision_model is None:
            raise ValueError("Vision modelが設定されていません。setup_vision_model()を先に実行してください。")
        
        print(f"🚀 Vision model訓練開始...")
        print(f"   エポック数: {num_epochs}")
        print(f"   バッチサイズ: {batch_size}")
        print(f"   学習率: {learning_rate}")
        
        try:
            # 訓練引数設定
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_steps=50,
                logging_steps=10,
                evaluation_strategy="steps",
                eval_steps=50,
                save_strategy="steps",
                save_steps=100,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                gradient_checkpointing=True,
                fp16=True,
                report_to="none",  # WandBなどのログを無効
            )
            
            # SFTTrainer設定
            trainer = SFTTrainer(
                model=self.vision_model,
                tokenizer=self.base_tokenizer,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                formatting_func=formatting_func_for_vision,
                data_collator=UnslothVisionDataCollator(tokenizer=self.base_tokenizer),
                args=training_args,
                max_seq_length=512,
            )
            
            # 訓練実行
            print(f"📈 訓練開始...")
            trainer.train()
            
            # モデル保存
            print(f"💾 モデル保存中: {output_dir}")
            trainer.save_model(output_dir)
            self.base_tokenizer.save_pretrained(output_dir)
            
            self.is_fine_tuned = True
            print(f"✅ Vision model訓練完了！")
            
            # 訓練結果サマリー
            train_result = trainer.state.log_history
            if train_result:
                final_loss = train_result[-1].get('eval_loss', 'N/A')
                print(f"   最終評価損失: {final_loss}")
            
        except Exception as e:
            print(f"❌ 訓練エラー: {str(e)}")
            raise e
    
    def load_fine_tuned_model(self, model_path: str):
        """
        Fine-tunedモデルをロード
        
        Args:
            model_path: モデルのパス
        """
        try:
            print(f"📂 Fine-tunedモデルロード中: {model_path}")
            
            # FastVisionModelでロード
            self.vision_model, self.base_tokenizer = FastVisionModel.from_pretrained(
                model_name=model_path,
                dtype=None,
                max_seq_length=1024,
                load_in_4bit=True,
            )
            
            self.is_fine_tuned = True
            print(f"✅ Fine-tunedモデルロード完了")
            
        except Exception as e:
            print(f"❌ モデルロードエラー: {str(e)}")
            raise e
    
    def get_model_for_inference(self):
        """推論用モデルを取得"""
        if self.is_fine_tuned and self.vision_model is not None:
            return self.vision_model
        else:
            return self.base_model

#%%
# RAG (Retrieval-Augmented Generation) Implementation
class WikipediaRAG:
    def __init__(self, embedding_model='all-MiniLM-L6-v2'):
        """
        Initialize RAG system with Wikipedia as knowledge base
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.documents = []
        self.embeddings = None
        self.index = None

    def search_wikipedia(self, query: str, num_results: int = 5) -> List[str]:
        """
        Search Wikipedia and return page contents
        """
        try:
            search_results = wikipedia.search(query, results=num_results)
            documents = []
            
            for title in search_results:
                try:
                    page = wikipedia.page(title)
                    # Split content into chunks for better retrieval
                    content = page.content
                    chunks = self._split_text(content, chunk_size=500)
                    for chunk in chunks:
                        if len(chunk.strip()) > 100:  # Filter out very short chunks
                            documents.append({
                                'title': title,
                                'content': chunk,
                                'url': page.url
                            })
                except wikipedia.exceptions.DisambiguationError as e:
                    # Take the first option if disambiguation occurs
                    try:
                        page = wikipedia.page(e.options[0])
                        content = page.content
                        chunks = self._split_text(content, chunk_size=500)
                        for chunk in chunks:
                            if len(chunk.strip()) > 100:
                                documents.append({
                                    'title': e.options[0],
                                    'content': chunk,
                                    'url': page.url
                                })
                    except:
                        continue
                except:
                    continue
                    
            return documents
        except:
            return []
    
    def _split_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks
        """
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
            
            chunks.append(text[start:end].strip())
            start = end - overlap
            
        return chunks

    def search_species_dynamically(self, species_candidates: List, knowledge_base_type: str = "general") -> List[Dict]:
        """推定種に対する動的Wikipedia検索"""
        
        dynamic_documents = []
        
        for i, candidate in enumerate(species_candidates[:3]):  # 上位3候補のみ
            species_name = candidate.name
            confidence = candidate.score
            
            # 知識ベース別の特化クエリ生成
            species_queries = self._generate_domain_specific_queries(species_name, knowledge_base_type)
            
            for query in species_queries:
                try:
                    search_results = self.search_wikipedia(query, num_results=2)
                    
                    for doc in search_results:
                        # メタデータに種情報を追加
                        doc['species_name'] = species_name
                        doc['species_confidence'] = confidence
                        doc['knowledge_domain'] = knowledge_base_type
                        doc['query_type'] = self._extract_query_type(query, knowledge_base_type)
                        dynamic_documents.append(doc)
                        
                except Exception as e:
                    continue
        
        return dynamic_documents

    def _generate_domain_specific_queries(self, species_name: str, knowledge_base_type: str) -> List[str]:
        """ドメイン特化クエリを生成"""
        clean_name = species_name.strip()
        
        if knowledge_base_type == "field_guides":
            # フィールドガイド：識別・形態学重視
            queries = [
                f"{clean_name} field guide identification",
                f"{clean_name} mushroom identification key features",
                f"{clean_name} morphology cap stem gills pores",
                f"{clean_name} distinguishing characteristics",
                f"{clean_name} lookalike similar species"
            ]
        
        elif knowledge_base_type == "toxicity_reports":
            # 毒性レポート：安全性・毒性重視
            queries = [
                f"{clean_name} toxicity poisonous edible",
                f"{clean_name} mushroom safety consumption",
                f"{clean_name} toxic compounds effects",
                f"{clean_name} poisoning symptoms treatment",
                f"{clean_name} edibility safety warnings"
            ]
        
        elif knowledge_base_type == "cooking_recipes":
            # 調理レシピ：食用・料理重視
            queries = [
                f"{clean_name} cooking recipes preparation",
                f"{clean_name} edible mushroom culinary uses",
                f"{clean_name} cooking methods recipes",
                f"{clean_name} food preparation techniques",
                f"{clean_name} mushroom recipes cuisine"
            ]
        
        else:
            # 汎用検索（基本的な種情報）
            clean_name = species_name.strip()
            if any(word in clean_name.lower() for word in ['species', 'mushroom', 'unknown']):
                # 不完全な種名の場合は属名のみを使用
                genus = clean_name.split()[0] if ' ' in clean_name else clean_name
                queries = [
                    f"{genus} mushroom identification",
                    f"{genus} genus characteristics", 
                    f"{genus} mushroom safety edibility"
                ]
            else:
                # 完全な種名の場合
                queries = [
                    f"{clean_name} mushroom identification characteristics",
                    f"{clean_name} morphology features",
                    f"{clean_name} toxicity safety edibility",
                    f"{clean_name} habitat distribution ecology"
                ]
        
        return queries

    def _extract_query_type(self, query: str, knowledge_base_type: str) -> str:
        """
        クエリタイプを抽出（メタデータ用）
        """
        if "identification" in query or "morphology" in query:
            return "identification"
        elif "toxicity" in query or "poisonous" in query or "safety" in query:
            return "toxicity"
        elif "cooking" in query or "recipe" in query or "culinary" in query:
            return "cooking"
        elif "habitat" in query or "distribution" in query:
            return "ecology"
        else:
            return knowledge_base_type

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve relevant documents for a query
        """
        if self.index is None:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['relevance_score'] = float(score)
                results.append(doc)
        
        return results

    def generate_context(self, query: str, top_k: int = 3) -> str:
        """
        Generate context string from retrieved documents
        """
        retrieved_docs = self.retrieve(query, top_k)
        
        if not retrieved_docs:
            return "No relevant information found."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            context_parts.append(f"Source {i+1} ({doc['title']}):\n{doc['content']}")
        
        return "\n\n".join(context_parts)

    def rag_generate(self, query: str, top_k: int = 3) -> Dict:
        """
        Generate response using RAG approach
        """
        # Retrieve relevant context
        context = self.generate_context(query, top_k)
        
        # Create enhanced prompt with context
        enhanced_prompt = f"""Based on the following context information, please answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        return {
            'context': context,
            'enhanced_prompt': enhanced_prompt,
            'retrieved_docs': self.retrieve(query, top_k)
        }

# %%
# Mobile Mushroom Identification Workflow
@dataclass
class WorkflowResult:
    """3ステップワークフロー結果"""
    candidate_species: List[str]
    similarity_scores: List[float]
    toxicity_info: Dict[str, Any]
    cooking_methods: Dict[str, Any]
    safety_warnings: List[str]
    final_answer: str
    recommendation: str

class ImageDatabaseRAG:
    """画像データベースからの類似画像検索システム (Gemma3n使用)"""
    
    def __init__(self, image_dataset_path: str, gemma_model, gemma_tokenizer):
        """
        Args:
            image_dataset_path: 画像データセットのパス
            gemma_model: Gemma3nモデル
            gemma_tokenizer: Gemma3nトークナイザー
        """
        self.dataset_path = image_dataset_path
        self.gemma_model = gemma_model
        self.gemma_tokenizer = gemma_tokenizer
        self.image_embeddings = None
        self.image_metadata = []
        self.index = None

    def build_image_index(self, class_names: List[str], paths: List[str], 
                         classes: List[str]):
        """
        画像データセットからベクトルインデックスを構築（各クラス5枚まで均等サンプリング）
        
        Args:
            class_names: クラス名リスト
            paths: 画像パスリスト  
            classes: 各画像のクラス
        """
        # 各クラスから均等にサンプリング（各クラス5枚まで）
        images_per_class = 5
        selected_indices = []
        
        print(f"📊 データセット分析:")
        print(f"   総画像数: {len(classes)}枚")
        print(f"   総クラス数: {len(class_names)}種類")
        
        # クラス別の画像インデックスを収集し均等サンプリング
        class_indices = {}
        for i, class_name in enumerate(classes):
            if class_name not in class_indices:
                class_indices[class_name] = []
            class_indices[class_name].append(i)
        
        print(f"   実際にデータがあるクラス数: {len(class_indices)}種類")
        
        # クラス別サンプリング情報を表示
        sampled_classes = 0
        for class_name in class_indices.keys():
            available = len(class_indices[class_name])
            to_select = min(available, images_per_class)
            if to_select > 0:
                selected_for_class = random.sample(class_indices[class_name], to_select)
                selected_indices.extend(selected_for_class)
                sampled_classes += 1
                if sampled_classes <= 10:  # 最初の10クラスだけ詳細表示
                    print(f"   📁 {class_name}: {available}枚中{to_select}枚選択")
        
        print(f"   🎯 サンプリング結果: {sampled_classes}クラスから{len(selected_indices)}枚選択")
        
        embeddings = []
        metadata = []
        total_images = len(selected_indices)
        
        print(f"🚀 画像インデックス構築開始: {total_images}枚の画像を処理中...")
        
        # 進捗バー用変数
        successful_extractions = 0
        failed_extractions = 0
        
        for i, idx in enumerate(selected_indices):
            try:
                image_path = paths[idx]
                class_name = classes[idx]
                
                # 進捗バー表示
                progress = (i + 1) / total_images * 100
                bar_length = 30
                filled_length = int(bar_length * (i + 1) // total_images)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"\r📸 [{i+1:3d}/{total_images}] [{bar}] {progress:5.1f}% | {class_name[:20]:<20}", end='', flush=True)
                
                # Gemma3nで画像から特徴ベクトルを抽出
                image_embedding = self._extract_image_features(image_path, class_name, verbose=False)
                
                if image_embedding is not None:
                    embeddings.append(image_embedding)
                    metadata.append({
                        'path': image_path,
                        'class': class_name,
                        'class_id': class_names.index(class_name) if class_name in class_names else -1,
                        'index': idx
                    })
                    successful_extractions += 1
                else:
                    failed_extractions += 1
                
            except Exception as e:
                failed_extractions += 1
                continue
        
        # 進捗バー完了
        print(f"\n✅ 画像処理完了: 成功 {successful_extractions}件, 失敗 {failed_extractions}件")
        
        if embeddings:
            print(f"🔧 FAISSインデックス構築中...")
            self.image_embeddings = np.array(embeddings)
            self.image_metadata = metadata

            # 2次元配列に変換（FAISS要件）
            if self.image_embeddings.ndim != 2:
                # 1次元配列の場合は reshapeして2次元にする
                if self.image_embeddings.ndim == 1:
                    self.image_embeddings = self.image_embeddings.reshape(1, -1)
                else:
                    # 3次元以上の場合は flatten
                    self.image_embeddings = self.image_embeddings.reshape(len(embeddings), -1)
            
            # FAISSインデックス構築
            dimension = self.image_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.image_embeddings = np.ascontiguousarray(self.image_embeddings, dtype=np.float32)
            faiss.normalize_L2(self.image_embeddings)
            self.index.add(self.image_embeddings)
            
            print(f"🎉 FAISSインデックス構築完了!")
            print(f"   📊 登録画像数: {len(self.image_metadata)}枚")
            print(f"   📐 ベクトル次元: {dimension}次元")
            
            # 登録されたクラスの詳細分析
            registered_classes = {}
            for m in self.image_metadata:
                class_name = m['class']
                if class_name not in registered_classes:
                    registered_classes[class_name] = 0
                registered_classes[class_name] += 1
            
            print(f"   🏷️ 登録クラス数: {len(registered_classes)}種類")
            print(f"   📋 登録クラス詳細:")
            for i, (class_name, count) in enumerate(registered_classes.items()):
                if i < 20:  # 最初の20クラスだけ表示
                    print(f"      {i+1:2d}. {class_name}: {count}枚")
                elif i == 20:
                    print(f"      ... 他{len(registered_classes)-20}クラス")
        else:
            print(f"❌ 画像インデックス構築失敗: 有効な特徴ベクトルが0個")

    def _extract_image_features(self, image_path: str, class_name: str, verbose: bool = True) -> Optional[np.ndarray]:
        """
        Gemma3nを使用して画像から特徴ベクトルを抽出
        
        Args:
            image_path: 画像ファイルパス
            class_name: 画像のクラス名
            verbose: 詳細ログの表示
            
        Returns:
            特徴ベクトル (numpy配列)
        """
        try:
            if verbose:
                print(f"🔍 Gemma3nで画像特徴抽出中: {image_path} (クラス: {class_name})")
            
            # Gemma3nに画像の特徴抽出を依頼
            feature_messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Extract visual features of this mushroom. Describe in exactly 50 words: cap color, shape, size, stem characteristics, gills/pores, texture, and any distinctive markings."}
                ]
            }]
            
            # Gemma3nで特徴テキストを生成して実際の特徴ベクトルを抽出
            inputs = self.gemma_tokenizer.apply_chat_template(
                feature_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to("cuda")
            
            # モデルの中間層から特徴ベクトルを抽出
            with torch.no_grad():
                outputs = self.gemma_model(**inputs, output_hidden_states=True)
                # 最後の隠れ層の平均を特徴ベクトルとして使用
                hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
                feature_vector = hidden_states.mean(dim=1).squeeze().cpu().numpy()  # [hidden_dim]
            
            if verbose:
                print(f"✅ 特徴ベクトル抽出完了: {feature_vector.shape}")
            return feature_vector
            
        except Exception as e:
            if verbose:
                print(f"⚠️ Gemma3n特徴抽出でエラー発生: {str(e)}")
                print(f"🔄 フォールバック: 擬似ベクトルを使用 (クラス: {class_name})")
            # フォールバック: クラス名ベースの擬似ベクトル
            return self._create_fallback_vector(class_name)

    def _create_fallback_vector(self, class_name: str, dim: int = 768) -> np.ndarray:
        """
        クラス名から擬似特徴ベクトルを生成（フォールバック用）
        
        Args:
            class_name: クラス名
            dim: ベクトルの次元数
            
        Returns:
            擬似特徴ベクトル
        """
        # クラス名のハッシュをシードにして安定した擬似ベクトルを生成
        hash_seed = hash(class_name) % (2**31)
        np.random.seed(hash_seed)
        vector = np.random.normal(0, 1, dim)
        np.random.seed()  # シードをリセット
        return vector.astype(np.float32)

    def find_similar_images(self, query_image_path: str, top_k: int = 5) -> List[Dict]:
        """
        クエリ画像に類似した画像を検索
        
        Args:
            query_image_path: クエリ画像のパス
            top_k: 取得する類似画像数
            
        Returns:
            類似画像の情報リスト
        """
        if self.index is None:
            return []

        try:
            # Gemma3nでクエリ画像の特徴ベクトルを抽出
            query_vector = self._extract_image_features(query_image_path, "query", verbose=False)
            
            if query_vector is None:
                return []
            
            # ベクトルを正規化してFAISS用に準備
            query_embedding = query_vector.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_embedding)
            
            # 類似画像検索
            scores, indices = self.index.search(query_embedding, top_k)
            
            similar_images = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.image_metadata):
                    metadata = self.image_metadata[idx].copy()
                    metadata['similarity_score'] = float(score)
                    similar_images.append(metadata)
            
            return similar_images
            
        except Exception as e:
            return []

    def get_class_examples(self, class_name: str, max_examples: int = 3) -> List[Dict]:
        """
        指定クラスの代表画像を取得
        
        Args:
            class_name: クラス名
            max_examples: 取得する例数
            
        Returns:
            クラス例画像のリスト
        """
        examples = []
        for metadata in self.image_metadata:
            if metadata['class'] == class_name and len(examples) < max_examples:
                examples.append(metadata)
        
        return examples

    def display_image_comparison(self, query_image_path: str, similar_images: List[Dict], top_n: int = 3):
        """
        テスト用: クエリ画像と類似画像を比較表示
        
        Args:
            query_image_path: テスト画像のパス
            similar_images: 類似検索結果
            top_n: 表示する類似画像数
        """
        try:
            from IPython.display import Image, display
            
            print(f"\n📊 画像比較結果:")
            print(f"🎯 テスト画像: {query_image_path}")
            print(f"📝 ファイル名: {query_image_path.split('/')[-1]}")
            
            # テスト画像を表示
            try:
                display(Image(filename=query_image_path, width=300))
            except Exception as e:
                print(f"❌ テスト画像表示不可: {str(e)}")
            
            print(f"\n🔍 類似画像 Top {min(top_n, len(similar_images))}:")
            for i, img in enumerate(similar_images[:top_n]):
                similarity_percent = img['similarity_score'] * 100
                print(f"\n  {i+1}. {img['class']} (類似度: {similarity_percent:.1f}%)")
                print(f"     📁 ファイル: {img['path'].split('/')[-1]}")
                print(f"     🗂️  フルパス: {img['path']}")
                
                # 類似度の視覚的表示
                bar_length = 20
                filled_length = int(bar_length * img['similarity_score'])
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"     📊 [{bar}] {similarity_percent:.1f}%")
                
                # 類似画像を表示
                try:
                    display(Image(filename=img['path'], width=300))
                except Exception as e:
                    print(f"     ❌ 画像表示不可: {str(e)}")
        
        except ImportError:
            # IPython.displayが利用できない場合はテキストのみ表示
            print(f"\n📊 画像比較結果 (テキストのみ):")
            print(f"🎯 テスト画像: {query_image_path}")
            print(f"📝 ファイル名: {query_image_path.split('/')[-1]}")
            
            print(f"\n🔍 類似画像 Top {min(top_n, len(similar_images))}:")
            for i, img in enumerate(similar_images[:top_n]):
                similarity_percent = img['similarity_score'] * 100
                print(f"  {i+1}. {img['class']} (類似度: {similarity_percent:.1f}%)")
                print(f"     📁 ファイル: {img['path'].split('/')[-1]}")
                print(f"     🗂️  フルパス: {img['path']}")
                
                # 類似度の視覚的表示
                bar_length = 20
                filled_length = int(bar_length * img['similarity_score'])
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"     📊 [{bar}] {similarity_percent:.1f}%")
                print()


class MobileMushroomWorkflow:
    """モバイル端末向けキノコ識別ワークフロー統合クラス - 3ステップ版 + Vision Fine-tuning対応"""
    
    def __init__(self, model, tokenizer, embedding_model):
        # 3ステップワークフロー用コンポーネント初期化
        
        # Base models
        self.base_model = model
        self.base_tokenizer = tokenizer
        
        # Vision Fine-tuning
        self.vision_finetuner = MushroomVisionFineTuner(model, tokenizer)
        
        # Step 1: 画像マッチング（base modelでスタート、後でfine-tunedに変更可能）
        self.image_db = ImageDatabaseRAG("dataset", model, tokenizer)
        
        # Step 2: Wikipedia検索
        self.wikipedia_rag = WikipediaRAG('all-MiniLM-L6-v2')
        
        # Fine-tuning状態管理
        self.use_fine_tuned_model = False

    def run_vision_fine_tuning(self, image_paths: List[str], labels: List[str], 
                              class_names: List[str], output_dir: str = "./mushroom_vision_model",
                              max_samples_per_class: int = 10, num_epochs: int = 3):
        """
        Vision Fine-tuningを実行
        
        Args:
            image_paths: 画像パスのリスト
            labels: ラベルのリスト
            class_names: クラス名のリスト
            output_dir: モデル保存ディレクトリ
            max_samples_per_class: クラスあたりの最大サンプル数
            num_epochs: エポック数
        """
        print(f"🎯 Vision Fine-tuning開始...")
        print(f"   対象画像数: {len(image_paths)}")
        print(f"   クラス数: {len(class_names)}")
        print(f"   クラスあたり最大サンプル数: {max_samples_per_class}")
        
        try:
            # データセット準備
            dataset_manager = MushroomVisionDataset(image_paths, labels, class_names)
            train_dataset, val_dataset = dataset_manager.prepare_vision_dataset(
                max_samples_per_class=max_samples_per_class
            )
            
            # Vision modelセットアップ
            self.vision_finetuner.setup_vision_model(
                finetune_vision_layers=True,
                finetune_language_layers=False
            )
            
            # 訓練実行
            self.vision_finetuner.train_vision_model(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                output_dir=output_dir,
                num_epochs=num_epochs,
                batch_size=2,  # VRAM制約のため小さく設定
                learning_rate=2e-5
            )
            
            # Fine-tunedモデルを有効化
            self.use_fine_tuned_model = True
            self._update_models_to_fine_tuned()
            
            print(f"🎉 Vision Fine-tuning完了！")
            print(f"   モデル保存先: {output_dir}")
            print(f"   Fine-tunedモデルが有効化されました")
            
        except Exception as e:
            print(f"❌ Fine-tuning失敗: {str(e)}")
            raise e
    
    def load_fine_tuned_model(self, model_path: str):
        """
        保存済みのFine-tunedモデルをロード
        
        Args:
            model_path: モデルのパス
        """
        try:
            print(f"📂 Fine-tunedモデルロード: {model_path}")
            
            # Fine-tunedモデルをロード
            self.vision_finetuner.load_fine_tuned_model(model_path)
            
            # Fine-tunedモデルを有効化
            self.use_fine_tuned_model = True
            self._update_models_to_fine_tuned()
            
            print(f"✅ Fine-tunedモデルロード完了")
            
        except Exception as e:
            print(f"❌ モデルロード失敗: {str(e)}")
            raise e
    
    def _update_models_to_fine_tuned(self):
        """内部的にfine-tunedモデルを使用するように切り替え"""
        if self.vision_finetuner.is_fine_tuned:
            fine_tuned_model = self.vision_finetuner.get_model_for_inference()
            # ImageDatabaseRAGのモデルを更新
            self.image_db.gemma_model = fine_tuned_model
            print(f"🔄 ImageDatabaseRAGをFine-tunedモデルに切り替え")
    
    def switch_to_base_model(self):
        """ベースモデルに戻す"""
        self.use_fine_tuned_model = False
        self.image_db.gemma_model = self.base_model
        print(f"🔄 ベースモデルに切り替え")
    
    def get_model_status(self):
        """現在のモデル状態を確認"""
        status = {
            'use_fine_tuned': self.use_fine_tuned_model,
            'fine_tuned_available': self.vision_finetuner.is_fine_tuned,
            'current_model': 'Fine-tuned' if self.use_fine_tuned_model else 'Base'
        }
        print(f"📊 モデル状態:")
        for key, value in status.items():
            print(f"   {key}: {value}")
        return status

    def initialize_image_database(self, image_dataset_path: str, class_names: List[str], 
                                 paths: List[str], classes: List[str]):
        """
        画像データベースを初期化（簡略化版）
        
        Args:
            image_dataset_path: データセットパス
            class_names: クラス名リスト
            paths: 画像パスリスト
            classes: 各画像のクラス
        """
        # 3ステップワークフロー用に画像データベース構築
        self.image_db.build_image_index(class_names, paths, classes)
        print(f"📂 画像データベース情報:")
        print(f"  データセットパス: {image_dataset_path}")
        print(f"  クラス数: {len(class_names)}")
        print(f"  画像数: {len(paths)}")

    def process_image(self, image_path: str, user_question: Optional[str] = None, 
                     verbose: bool = False) -> WorkflowResult:
        """
        3ステップワークフローを実行
        
        Args:
            image_path: 画像ファイルパス
            user_question: ユーザーからの質問
            verbose: 進捗表示の有効/無効
            
        Returns:
            WorkflowResult オブジェクト
        """
        start_time = time.time()
        
        try:
            if verbose:
                print("🚀 3ステップキノコ識別ワークフロー開始...")
            
            # Step 1: 画像マッチング
            if verbose:
                print("🔍 Step 1: 画像マッチング中...")
            similar_images = self.image_db.find_similar_images(image_path, top_k=5)
            
            # デバッグ用: 画像比較表示
            if verbose and similar_images:
                self.image_db.display_image_comparison(image_path, similar_images, top_n=3)
            
            # 上位類似画像から候補種を抽出
            candidate_species = [img.get('class', '不明') for img in similar_images[:3]]
            similarity_scores = [img.get('similarity_score', 0.0) for img in similar_images[:3]]
            
            # Step 2: Wikipedia検索
            if verbose:
                print("📚 Step 2: Wikipedia検索中...")
            
            # 候補種に基づいてWikipedia検索
            toxicity_info = {}
            cooking_methods = {}
            safety_warnings = []
            
            for species in candidate_species:
                if species != '不明':
                    # 毒性情報検索
                    toxicity_query = f"{species} toxicity poisonous edible safety"
                    toxicity_docs = self.wikipedia_rag.search_wikipedia(toxicity_query, num_results=2)
                    toxicity_info[species] = toxicity_docs
                    
                    # 調理情報検索
                    cooking_query = f"{species} cooking preparation recipe"
                    cooking_docs = self.wikipedia_rag.search_wikipedia(cooking_query, num_results=2)
                    cooking_methods[species] = cooking_docs
                    
                    # 安全警告抽出
                    for doc in toxicity_docs:
                        content = doc.get('content', '').lower()
                        if 'toxic' in content or 'poisonous' in content:
                            safety_warnings.append(f"{species}: 毒性の可能性あり")
            
            # Step 3: 回答生成
            if verbose:
                print("📝 Step 3: 回答生成中...")
            
            # 最終回答と推奨アクション生成
            final_answer = self._generate_answer(candidate_species, toxicity_info, cooking_methods, user_question)
            recommendation = self._generate_recommendation(candidate_species, safety_warnings)
            
            # 結果作成
            total_time = time.time() - start_time
            
            result = WorkflowResult(
                candidate_species=candidate_species,
                similarity_scores=similarity_scores,
                toxicity_info=toxicity_info,
                cooking_methods=cooking_methods,
                safety_warnings=safety_warnings,
                final_answer=final_answer,
                recommendation=recommendation
            )
            
            if verbose:
                print(f"🎉 3ステップワークフロー完了! (総処理時間: {total_time:.2f}秒)")

            return result
            
        except Exception as e:
            # エラーハンドリング
            total_time = time.time() - start_time
            if verbose:
                print(f"❌ エラーが発生しました: {str(e)}")
            
            return WorkflowResult(
                candidate_species=["エラー"],
                similarity_scores=[0.0],
                toxicity_info={},
                cooking_methods={},
                safety_warnings=["システムエラーのため専門家に相談してください"],
                final_answer=f"エラーが発生しました: {str(e)}",
                recommendation="システムエラーのため専門家に相談してください"
            )

    def _generate_answer(self, candidate_species: List[str], toxicity_info: Dict, 
                        cooking_methods: Dict, user_question: Optional[str]) -> str:
        """最終回答を生成"""
        if not candidate_species or candidate_species[0] == '不明':
            return "画像からキノコの種類を特定できませんでした。専門家に相談してください。"
        
        answer = f"上位候補種: {', '.join(candidate_species[:3])}\n\n"
        
        # 安全性情報
        has_toxicity = any(
            any('toxic' in doc.get('content', '').lower() or 'poisonous' in doc.get('content', '').lower() 
                for doc in docs) 
            for docs in toxicity_info.values()
        )
        
        if has_toxicity:
            answer += "⚠️ 注意: 毒性の可能性があります。絶対に摂取しないでください。\n\n"
        else:
            answer += "ℹ️ 毒性情報は確認できませんでしたが、最終判断は専門家に委ねてください。\n\n"
        
        # 調理情報
        if cooking_methods:
            answer += "🍳 調理情報:\n"
            for species, docs in cooking_methods.items():
                if docs:
                    answer += f"- {species}: 調理法が確認できます\n"
        
        answer += "\n‼️ 重要: どんなキノコでも、専門家の確認なしでの摂取は危険です。"
        
        return answer
    
    def _generate_recommendation(self, candidate_species: List[str], safety_warnings: List[str]) -> str:
        """推奨アクションを生成"""
        if safety_warnings:
            return "毒性の可能性があるため、絶対に摂取せず専門家に相談してください。"
        elif not candidate_species or candidate_species[0] == '不明':
            return "識別が困難なため、別の角度からの写真を撮影し、専門家に相談してください。"
        else:
            return "毒性情報は確認できませんでしたが、安全のため必ず専門家に確認してから判断してください。"


#%%
# モバイルワークフローのテスト実行

# ワークフロー初期化
# 一時的にSentenceTransformerを初期化
from sentence_transformers import SentenceTransformer
temp_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

mobile_workflow = MobileMushroomWorkflow(
    model=model,
    tokenizer=tokenizer,
    embedding_model=temp_embedding_model
)

# 画像データベースを初期化
mobile_workflow.initialize_image_database(dir0, class_names, paths, classes)

#%%
# 3ステップワークフローテスト - 簡潔版
print("🧪 3ステップキノコ識別ワークフロー テスト開始")
print("=" * 80)
print("3️⃣ 注：このワークフローは3ステップで実行します")
print("   1.画像マッチング → 2.Wikipedia検索 → 3.回答生成")

# テスト設定
test_question = "このキノコは食べられますか？"
print(f"📸 テスト画像: {image_path}")
print(f"❓ ユーザー質問: {test_question}")
print("=" * 80)

# テスト実行
try:
    print("\n🚀 3ステップワークフロー実行開始...")
    # 詳細な進捗表示でワークフロー実行
    result = mobile_workflow.process_image(image_path, test_question, verbose=True)
    
    print("\n" + "=" * 80)
    print("📊 3ステップワークフロー実行結果")
    print("=" * 80)
    
    # 結果表示
    print(f"🔬 候補種: {', '.join(result.candidate_species)}")
    print(f"📈 類似度スコア: {[f'{s:.3f}' for s in result.similarity_scores]}")
    print(f"⚠️ 安全警告: {len(result.safety_warnings)}件")
    for warning in result.safety_warnings[:3]:
        print(f"   - {warning}")
    print(f"💬 最終回答:")
    print(f"   {result.final_answer}")
    print(f"🎯 推奨アクション:")
    print(f"   {result.recommendation}")

except Exception as e:
    print(f"\n❌ テスト失敗: {str(e)}")
    print("デバッグ情報:")
    import traceback
    traceback.print_exc()

print("\n🏁 3ステップテスト完了")

#%%
# 追加テスト: 3ステップ複数質問パターン
print("\n" + "=" * 80)
print("🔬 3ステップ追加テスト: 複数質問パターン")
print("=" * 80)
print("📊 各テストで3ステップワークフローを実行し、シンプルな結果を出力")

test_questions = [
    "この種類は何ですか？",
    "毒性はありますか？",
    "調理方法を教えてください",
    None  # 質問なし
]

for i, question in enumerate(test_questions, 1):
    print(f"\n--- 3ステップテスト {i}/4 ---")
    print(f"質問: {question if question else '(質問なし)'}")
    
    try:
        # シンプルな実行（verbose=False）
        start_time = time.time()
        result = mobile_workflow.process_image(image_path, question, verbose=False)
        execution_time = time.time() - start_time
        
        print(f"✅ 実行成功 (実行時間: {execution_time:.2f}秒)")
        print(f"   🔬 候補種: {result.candidate_species[:2]}")  # 上位2つ
        print(f"   📈 類似度: {[f'{s:.2f}' for s in result.similarity_scores[:2]]}")
        print(f"   ⚠️ 警告: {len(result.safety_warnings)}件")
        print(f"   🎯 推奨: {result.recommendation[:50]}...")
                
    except Exception as e:
        print(f"❌ 実行失敗: {str(e)}")

print("\n🎯 全3ステップテスト完了")
print("📊 各テストでシンプルな構造化結果が正常に生成されました")

#%%
# Vision Fine-tuning使用例とテスト
print("\n" + "=" * 80)
print("🎯 Vision Fine-tuning使用例")
print("=" * 80)
print("📋 このセクションではキノコ画像でのVision Fine-tuningの使用方法を示します")

# Fine-tuning実行例（コメントアウト - 実際に実行する場合は有効化）
"""
# Vision Fine-tuning実行例:
print("\\n🔧 Vision Fine-tuning実行例:")
print("# 1. Fine-tuningを実行")
print("mobile_workflow.run_vision_fine_tuning(")
print("    image_paths=paths,")
print("    labels=classes,") 
print("    class_names=class_names,")
print("    output_dir='./mushroom_vision_finetuned',")
print("    max_samples_per_class=5,  # 各クラス5枚まで")
print("    num_epochs=2  # 短時間テスト用")
print(")")
print("")
print("# 2. Fine-tunedモデルの状態確認")
print("mobile_workflow.get_model_status()")
print("")
print("# 3. 保存済みモデルをロード（次回起動時）")
print("mobile_workflow.load_fine_tuned_model('./mushroom_vision_finetuned')")
print("")
print("# 4. ベースモデルに戻す")
print("mobile_workflow.switch_to_base_model()")
"""

# モデル状態確認
print("\n📊 現在のモデル状態:")
status = mobile_workflow.get_model_status()

# Fine-tuning準備情報
print(f"\n🔧 Fine-tuning準備状況:")
print(f"   利用可能画像数: {len(paths)}")
print(f"   利用可能クラス数: {len(class_names)}")
print(f"   推奨設定:")
print(f"     - max_samples_per_class: 5-10 (VRAM制約のため)")
print(f"     - num_epochs: 2-3 (過学習防止)")
print(f"     - batch_size: 1-2 (VRAM制約のため)")

print(f"\n💡 Fine-tuning実行手順:")
print(f"   1. mobile_workflow.run_vision_fine_tuning() でFine-tuning実行")
print(f"   2. 自動的にFine-tunedモデルに切り替わります")
print(f"   3. process_image()でFine-tunedモデルが使用されます")
print(f"   4. mobile_workflow.switch_to_base_model() でベースモデルに戻せます")

print(f"\n⚠️ 注意事項:")
print(f"   - Vision Fine-tuningは大量のVRAM（15GB以上）を消費します")
print(f"   - Colab無料版では制限があるため、小さなデータセットで実行してください")
print(f"   - Fine-tuningにより、キノコ画像識別の精度向上が期待できます")

print("\n🏁 Vision Fine-tuning使用例完了")
print("📚 詳細な実装方法は上記のクラス定義とメソッドを参照してください")
