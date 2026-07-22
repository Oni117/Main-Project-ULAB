# CoAtNet Project - AI Coding Guide

## Project Overview
Medical image classification system using ResNet-50 with CBAM (Convolutional Block Attention Module) to distinguish between diabetic and non-diabetic retinal images. Single-file PyTorch implementation training on ImageFolder dataset structure.

## Architecture & Key Components

### Model Pipeline (main.py - lines 50-67)
- **Backbone**: ResNet-50 (ImageNet pretrained) with feature extraction at layer4
- **Attention**: Custom CBAM module applied post-conv layers for channel + spatial refinement
- **Classification Head**: Single Linear layer from 2048→num_classes
- **Key Pattern**: All custom modules inherit `nn.Module` and implement forward(); attention mechanisms always return scaled tensors (x * attention_weights)

### Attention Mechanism (CBAM, lines 18-47)
Two-stage attention applied sequentially:
1. **Channel Attention** (lines 26-31): Adaptive pooling → FC bottleneck → Sigmoid activation; multiplied element-wise
2. **Spatial Attention** (lines 33-36): Max+Mean concat → 7×7 conv → Sigmoid; same multiplication pattern
- **Critical Detail**: Spatial attention requires 2-channel input from max/mean concatenation (line 45)
- **Common Mistake**: Forgetting to scale feature maps after attention (always use `x * attention` not just attention output)

## Data & Training Workflow

### Dataset Structure (expects ImageFolder format)
```
dataset/
  diabetes/       (positive class - retinal images with diabetic indicators)
  nondiabetes/    (negative class - healthy retinal images)
```
- Auto-loads via `torchvision.datasets.ImageFolder`
- Supports any image format; resized to 224×224 (ImageNet standard)
- Class order: alphabetical (diabetes=0, nondiabetes=1)

### Data Split Strategy (lines 155-163)
- **60% train**, **20% val**, **20% test** via stratified splits
- **Stratified**: Maintains class ratios across splits using `StratifiedKFold`
- **Critical**: Use `train_test_split` twice with `stratify` parameter to avoid class imbalance
- Seeds fixed (random_state=42) for reproducibility

### Training Loop (lines 165-180)
- Hyperparameters hardcoded: BATCH_SIZE=16, EPOCHS=70, LR=1e-4
- Loss: `CrossEntropyLoss` (expects class indices, not one-hot)
- Optimizer: Adam (adaptive learning rate, good default for CNNs)
- **Validation Gap Metric**: (train_acc - val_acc) monitored each epoch to detect overfitting

### Cross-Validation (lines 210-243)
- 5-Fold stratified cross-validation **after** main training (not for hyperparameter tuning)
- Only 5 epochs per fold (quick validation check, not full retraining)
- Computes mean ± std of fold accuracies as robustness metric

## Development & Execution

### Running the Pipeline
```bash
python main.py
```
- Requires: torch, torchvision, sklearn, numpy, matplotlib
- Auto-detects GPU via CUDA availability; forces GPU device 0 in env vars (line 4)
- Outputs: Classification report, confusion matrix, 4 training curves, k-fold stats

### CUDA Configuration (line 4)
- Forces single GPU device (`"0"`) to avoid device conflicts
- Comment out if CPU-only or multi-GPU setup needed

### Key Outputs Generated
- **Plots**: loss_curve.png, accuracy_curve.png, generalization_curve.png, kfold_accuracy.png
- **Model File**: best_coatnet.pth (not auto-saved; manually add checkpointing if needed)
- **Reports**: classification_report.txt, optimization_report.txt (user-generated, not auto-created)

## Project-Specific Patterns

### Function Organization
- Modular: `train_one_epoch()`, `evaluate()` separately for reusability
- Both return structured data: losses (list), predictions (array), labels (array)
- Evaluation extends training with ground truth labels for metrics computation

### Device Handling
- Single device line: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- All tensors explicitly moved: `.to(device)` on inputs, outputs auto-on-device
- No DDP (distributed) setup; single GPU/CPU only

### Model Saving (Missing - Add if extending)
Current code lacks checkpoint saving. Pattern to use:
```python
torch.save(model.state_dict(), 'best_coatnet.pth')  # in training loop if val_acc improves
```

## Common Modifications

### Adjust Hyperparameters
Edit lines 142-144: `BATCH_SIZE`, `EPOCHS`, `LR`

### Change Model Backbone
Replace line 60 `models.resnet50()` with alternatives (resnet101, efficientnet, vit) - keep CBAM + fc head structure

### Add Early Stopping
Modify main training loop (lines 165-180) to track best_val_acc and break if no improvement for N epochs

### Extend Dataset
Add more image classes in dataset/ folder; auto-detects via `dataset.classes`

### Disable CBAM
Comment out line 76 (`self.cbam(x)`) or remove module entirely - baseline ResNet50 performance comparison

## Testing & Validation Checks
- Perfect accuracy (1.0) on classification_report.txt suggests small dataset or potential data leakage
- Verify train/val/test splits don't overlap (stratified split logic at lines 155-163 prevents this)
- Check dataset folder structure matches ImageFolder expectations (exactly 2 subfolders)
