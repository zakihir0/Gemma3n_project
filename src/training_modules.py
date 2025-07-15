"""
Training Modules
Vision Fine-tuning関連の機能
"""

import random
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datasets import Dataset
from PIL import Image
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

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
                    # Vision Fine-tuning用のメッセージ形式
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": path},
                            {"type": "text", "text": f"What type of mushroom is this? Identify this mushroom."}
                        ]
                    }, {
                        "role": "assistant", 
                        "content": f"This is a {class_name} mushroom."
                    }]
                    
                    selected_data.append({
                        'image_path': path,
                        'class_name': class_name,
                        'class_id': self.label_to_id.get(class_name, -1),
                        'messages': messages,
                        'text': f"This is a {class_name} mushroom."  # バックアップ用
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
    # メッセージ形式のデータをそのまま返す
    return examples

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
            # Training用にモデルを有効化（重要）
            FastVisionModel.for_training(self.vision_model)
            
            print(f"✅ Vision model設定完了")
            print(f"   Vision層 fine-tuning: {finetune_vision_layers}")
            print(f"   Language層 fine-tuning: {finetune_language_layers}")
            print(f"   Training mode有効化: ✅")
            
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
            # 訓練引数設定（SFTConfig使用）
            training_args = SFTConfig(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=4,  # メモリ効率化
                learning_rate=learning_rate,
                warmup_steps=50,
                logging_steps=10,
                eval_strategy="steps",
                eval_steps=50,
                save_strategy="steps",
                save_steps=100,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                # Vision特有の設定
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                dataloader_pin_memory=False,
                fp16=True,
                optim="adamw_8bit",  # メモリ効率化
                weight_decay=0.01,
                lr_scheduler_type="linear",
                report_to="none",  # WandBなどのログを無効
            )
            
            # SFTTrainer設定
            trainer = SFTTrainer(
                model=self.vision_model,
                tokenizer=self.base_tokenizer,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=UnslothVisionDataCollator(model=self.vision_model, processor=self.base_tokenizer),
                args=training_args,
                max_seq_length=512,
                # Vision用の特別設定
                dataset_text_field="messages",  # メッセージフィールドを指定
                packing=False,  # Vision用にパッキングを無効化
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

def run_mushroom_vision_finetuning(image_paths: List[str], labels: List[str], 
                                  class_names: List[str], base_model, base_tokenizer,
                                  output_dir: str = "./mushroom_vision_finetuned",
                                  max_samples_per_class: int = 5, num_epochs: int = 2):
    """
    キノコ画像でのVision Fine-tuningを独立実行
    
    Args:
        image_paths: 画像パスのリスト
        labels: ラベルのリスト
        class_names: クラス名のリスト
        base_model: ベースGemma3nモデル
        base_tokenizer: ベーストークナイザー
        output_dir: モデル保存ディレクトリ
        max_samples_per_class: クラスあたりの最大サンプル数
        num_epochs: エポック数
    
    Returns:
        Fine-tunedモデルとトークナイザー
    """
    print(f"🚀 キノコ画像Vision Fine-tuning開始...")
    print(f"   対象画像数: {len(image_paths)}")
    print(f"   クラス数: {len(class_names)}")
    print(f"   クラスあたり最大サンプル数: {max_samples_per_class}")
    print(f"   エポック数: {num_epochs}")
    print(f"   保存先: {output_dir}")
    
    try:
        # Step 1: データセット準備
        print(f"\n📊 Step 1: データセット準備")
        dataset_manager = MushroomVisionDataset(image_paths, labels, class_names)
        train_dataset, val_dataset = dataset_manager.prepare_vision_dataset(
            max_samples_per_class=max_samples_per_class
        )
        
        # Step 2: Fine-tunerセットアップ
        print(f"\n🔧 Step 2: Fine-tuner初期化")
        finetuner = MushroomVisionFineTuner(base_model, base_tokenizer)
        finetuner.setup_vision_model(
            finetune_vision_layers=True,
            finetune_language_layers=False
        )
        
        # Step 3: 訓練実行
        print(f"\n📈 Step 3: 訓練実行")
        finetuner.train_vision_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=1,  # VRAM制約のため1に設定
            learning_rate=2e-5
        )
        
        print(f"\n🎉 Vision Fine-tuning完了！")
        print(f"   保存先: {output_dir}")
        
        return finetuner.get_model_for_inference(), base_tokenizer
        
    except Exception as e:
        print(f"\n❌ Fine-tuning失敗: {str(e)}")
        raise e

def create_workflow_with_finetuned_model(finetuned_model, tokenizer, embedding_model):
    """
    Fine-tunedモデルを使用してワークフローを作成
    
    Args:
        finetuned_model: Fine-tunedモデル
        tokenizer: トークナイザー
        embedding_model: Embedingモデル
        
    Returns:
        Fine-tunedモデルを使用するワークフロー
    """
    print(f"🔄 Fine-tunedモデルでワークフロー作成中...")
    
    # Fine-tunedモデルを使用してワークフロー作成
    from .rag_workflow import MobileMushroomWorkflow
    
    workflow = MobileMushroomWorkflow(
        model=finetuned_model,
        tokenizer=tokenizer,
        embedding_model=embedding_model
    )
    
    print(f"✅ Fine-tunedワークフロー作成完了")
    return workflow