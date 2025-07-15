"""
Gemma3n Modular Package
機械学習とキノコ識別システムのモジュール化パッケージ
"""

# Core model functions
from .gemma3n_core import (
    initialize_model,
    do_gemma_3n_inference,
    load_preprocessed_data,
    process_dataset,
    get_model_tokenizer
)

# Training modules
from .training_modules import (
    MushroomVisionDataset,
    MushroomVisionFineTuner,
    run_mushroom_vision_finetuning,
    create_workflow_with_finetuned_model,
    formatting_func_for_vision
)

# RAG and workflow
from .rag_workflow import (
    WikipediaRAG,
    ImageDatabaseRAG,
    MobileMushroomWorkflow,
    WorkflowResult
)

__version__ = "1.0.0"
__author__ = "Gemma3n Team"

# Main export lists for convenience
__all__ = [
    # Core functions
    "initialize_model",
    "do_gemma_3n_inference", 
    "load_preprocessed_data",
    "process_dataset",
    "get_model_tokenizer",
    
    # Training classes and functions
    "MushroomVisionDataset",
    "MushroomVisionFineTuner",
    "run_mushroom_vision_finetuning",
    "create_workflow_with_finetuned_model",
    "formatting_func_for_vision",
    
    # RAG and workflow classes
    "WikipediaRAG",
    "ImageDatabaseRAG", 
    "MobileMushroomWorkflow",
    "WorkflowResult"
]

def get_package_info():
    """パッケージ情報を取得"""
    return {
        "name": "gemma3n",
        "version": __version__,
        "author": __author__,
        "modules": {
            "gemma3n_core": "モデル初期化・インファレンス・データ処理",
            "training_modules": "Vision Fine-tuning関連機能",
            "rag_workflow": "RAG・画像検索・モバイルワークフロー"
        },
        "classes": len([x for x in __all__ if x[0].isupper()]),
        "functions": len([x for x in __all__ if x[0].islower()])
    }

# パッケージ情報表示
def show_package_info():
    """パッケージ情報を表示"""
    info = get_package_info()
    print(f"📦 {info['name']} v{info['version']}")
    print(f"👨‍💻 Author: {info['author']}")
    print(f"📁 Modules: {len(info['modules'])}")
    for module, desc in info['modules'].items():
        print(f"  - {module}: {desc}")
    print(f"🔧 Classes: {info['classes']}, Functions: {info['functions']}")