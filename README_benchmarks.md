# Mushroom Classification Benchmark Suite

This benchmark suite compares three different approaches for mushroom classification:

1. **Gemma3n Vision Fine-tuning** (multimodal language model)
2. **CNN (ResNet50)** (traditional computer vision)
3. **Vision Transformer** (modern transformer-based vision)

## 📁 Files Overview

### Core Training Scripts
- `gemma3n.py` - Original Gemma3n vision fine-tuning implementation
- `cnn_benchmark.py` - CNN (ResNet50) training for comparison
- `vit_benchmark.py` - Vision Transformer training for comparison

### Analysis and Comparison
- `benchmark_comparison.py` - Performance comparison and visualization
- `run_benchmarks.py` - Automated benchmark execution script

### Generated Results
- `best_cnn_model.pth` - Best CNN model checkpoint
- `best_vit_model.pth` - Best Vision Transformer model checkpoint
- `*_confusion_matrix.png` - Confusion matrices for each model
- `*_training_curves.png` - Training/validation curves
- `benchmark_comparison.png` - Performance comparison charts
- `benchmark_results.json` - Detailed benchmark results

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install timm wandb scikit-learn matplotlib seaborn tqdm
```

### Run All Benchmarks
```bash
python run_benchmarks.py
```

### Run Individual Benchmarks
```bash
# CNN Benchmark
python cnn_benchmark.py

# Vision Transformer Benchmark  
python vit_benchmark.py

# Generate Comparison
python benchmark_comparison.py
```

## 📊 Model Comparison

### CNN (ResNet50)
- **Approach**: Convolutional Neural Network with ResNet50 backbone
- **Batch Size**: 32
- **Optimizer**: Adam (lr=0.001)
- **Advantages**: Fast training, efficient inference, well-established
- **Use Case**: Production deployment, resource-constrained environments

### Vision Transformer
- **Approach**: Transformer architecture with patch-based processing
- **Model**: ViT-Base (patch size 16, image size 224)
- **Batch Size**: 16 (due to memory requirements)
- **Optimizer**: AdamW (lr=1e-4)
- **Advantages**: State-of-the-art performance, attention mechanisms
- **Use Case**: Research, highest accuracy requirements

### Gemma3n Vision Fine-tuning
- **Approach**: Multimodal language model with vision capabilities
- **Features**: Natural language output, multimodal understanding
- **Advantages**: Flexible output format, transfer learning
- **Use Case**: Multimodal applications, descriptive outputs

## 🎯 Performance Metrics

The benchmark evaluates models on:
- **Accuracy**: Classification accuracy on validation set
- **Training Time**: Time to complete training
- **Model Size**: Number of parameters
- **Inference Speed**: Time per prediction
- **Memory Usage**: GPU memory requirements

## 📈 Expected Results

| Model | Accuracy | Training Time | Model Size | Best For |
|-------|----------|---------------|------------|----------|
| CNN (ResNet50) | ~82% | Fast | 25M params | Production |
| Vision Transformer | ~85% | Medium | 86M params | Research |
| Gemma3n | ~75% | Slow | 3B params | Multimodal |

*Note: Actual results may vary depending on dataset and hardware*

## 🔧 Configuration

### Dataset Configuration
- **Dataset Path**: `/notebooks/kaggle/input/mushroom1/merged_dataset`
- **Train/Test Split**: 80/20
- **Data Augmentation**: Random flip, rotation, color jitter
- **Image Size**: 224x224 pixels

### WandB Integration
All benchmarks log to Weights & Biases for experiment tracking:
- Project: `mushroom-classification-benchmark`
- Tracks: loss, accuracy, learning rate, confusion matrices
- Visualizations: training curves, sample predictions

## 📋 Usage Examples

### Training a CNN Model
```python
from cnn_benchmark import MushroomCNN, train_epoch, validate

# Initialize model
model = MushroomCNN(num_classes=10)

# Train
train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
```

### Using Vision Transformer
```python
from vit_benchmark import MushroomViT

# Initialize ViT model
model = MushroomViT(num_classes=10, model_name='vit_base_patch16_224')
```

### Generating Comparisons
```python
from benchmark_comparison import create_performance_comparison

# Create comparison charts
create_performance_comparison()
```

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in benchmark scripts
   - Use gradient accumulation for effective larger batch sizes

2. **Missing Dataset**
   - Ensure dataset path is correct in scripts
   - Check that mushroom_dataset.json exists

3. **WandB Login Issues**
   - Set WANDB_API_KEY environment variable
   - Run `wandb login` before executing scripts

### Performance Tips

1. **For Faster Training**
   - Use mixed precision training (fp16/bf16)
   - Reduce image resolution for initial experiments
   - Use smaller models for prototyping

2. **For Better Accuracy**
   - Increase training epochs
   - Use better data augmentation
   - Implement learning rate scheduling

## 📊 Interpreting Results

### Confusion Matrix
- Shows per-class classification performance
- Diagonal elements indicate correct predictions
- Off-diagonal elements show misclassifications

### Training Curves
- Monitor for overfitting (validation loss increases)
- Check convergence (loss stabilizes)
- Verify learning rate appropriateness

### Accuracy Metrics
- Overall accuracy: correct predictions / total predictions
- Per-class precision and recall in classification report
- F1-score for balanced evaluation

## 🎉 Next Steps

1. **Analyze Results**: Review WandB dashboard and generated charts
2. **Optimize Models**: Tune hyperparameters based on results
3. **Deploy Best Model**: Choose based on use case requirements
4. **Extend Evaluation**: Add more metrics (inference time, memory usage)
5. **Compare with Gemma3n**: Update comparison with actual results

## 📞 Support

For issues or questions:
- Check the troubleshooting section above
- Review WandB logs for training details
- Examine generated result files for debugging

Happy benchmarking! 🚀