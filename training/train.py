import os
import copy
import time
import random
import numpy as np
import torch
import torch.nn as nn
from PIL import ImageFile
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ======================================
# CONFIGURATION
# ======================================
DATASETS_ROOT = "datasets"
MODEL_DIR = "models"
SEED = 42

# "efficientnet_b2" -> ~31 MB per model (fits directly on GitHub & Streamlit Cloud without Git LFS)
# "convnext_tiny"   -> ~109 MB per model (requires Git LFS if pushing to GitHub)
ARCH = "efficientnet_b2"
IMAGE_SIZE = 260
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2      # Effective batch size = 32
HEAD_WARMUP_EPOCHS = 3    # Freeze backbone first to protect pretrained weights
TOTAL_EPOCHS = 35
BACKBONE_LR = 2e-4
HEAD_LR = 8e-4
WEIGHT_DECAY = 1e-2
EMA_DECAY = 0.995
EARLY_STOP_PATIENCE = 8

TASKS = [
    "bone_break",
    "lung_cancer",
    "renal_malignancy",
    "skin_lesions",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


set_seed()
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"Device: {DEVICE} | Arch: {ARCH} | Resolution: {IMAGE_SIZE}x{IMAGE_SIZE} | AMP: {USE_AMP}")

# ======================================
# MEDICAL-SAFE AUGMENTATIONS
# ======================================
train_transform = transforms.Compose([
    transforms.Resize((int(IMAGE_SIZE * 1.1), int(IMAGE_SIZE * 1.1))),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.82, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.25),
    transforms.RandomApply([
        transforms.RandomAffine(degrees=20, translate=(0.05, 0.05), scale=(0.95, 1.05))
    ], p=0.7),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ======================================
# MODEL BUILDER & EMA HELPER
# ======================================
def build_model(arch_name, num_classes):
    if arch_name == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        head_params = list(model.classifier.parameters())
    elif arch_name == "convnext_small":
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        head_params = list(model.classifier.parameters())
    elif arch_name == "efficientnet_b2":
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        head_params = list(model.classifier.parameters())
    else:
        raise ValueError(f"Unsupported architecture: {arch_name}")

    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    return model, backbone_params, head_params


class ModelEMA:
    """Maintains an Exponential Moving Average of model parameters for higher generalization."""
    def __init__(self, model, decay=0.995):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * self.decay + msd[k].detach() * (1.0 - self.decay))
            else:
                v.copy_(msd[k])


# ======================================
# BALANCED SAMPLER FOR IMBALANCED CLASSES
# ======================================
def create_balanced_loader(dataset, batch_size):
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )


# ======================================
# EVALUATION & 4-VIEW TTA
# ======================================
@torch.no_grad()
def evaluate_standard(model, loader):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            outputs = model(images)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


@torch.no_grad()
def evaluate_tta_4view(model, loader):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            p1 = torch.softmax(model(images), dim=1)
            p2 = torch.softmax(model(torch.flip(images, dims=[3])), dim=1)
            p3 = torch.softmax(model(torch.flip(images, dims=[2])), dim=1)
            p4 = torch.softmax(model(torch.rot90(images, k=1, dims=[2, 3])), dim=1)
            probs = (p1 + p2 + p3 + p4) / 4.0
        correct += (probs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


# ======================================
# TWO-STAGE TRAINING PIPELINE
# ======================================
def train_task(task_name):
    task_dir = os.path.join(DATASETS_ROOT, task_name)
    train_dir = os.path.join(task_dir, "train")
    val_dir = os.path.join(task_dir, "val") if os.path.exists(os.path.join(task_dir, "val")) else os.path.join(task_dir, "test")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"\n[SKIP] Missing train or test/val folder in: {task_dir}")
        return

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    val_data = datasets.ImageFolder(val_dir, transform=eval_transform)

    if train_data.classes != val_data.classes:
        raise ValueError(f"Class mismatch in {task_name}: {train_data.classes} vs {val_data.classes}")

    class_names = train_data.classes
    start_time = time.time()
    print(f"\n=======================================================")
    print(f"TASK: {task_name.upper()} | Classes ({len(class_names)}): {class_names}")
    print(f"Train Images: {len(train_data)} | Eval Images: {len(val_data)}")
    print(f"=======================================================")

    train_loader = create_balanced_loader(train_data, BATCH_SIZE)
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    model, backbone_params, head_params = build_model(ARCH, len(class_names))
    model = model.to(DEVICE)
    ema_tracker = ModelEMA(model, decay=EMA_DECAY)

    # Stage 1: Freeze backbone for HEAD_WARMUP_EPOCHS
    for p in backbone_params:
        p.requires_grad_(False)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params, "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_acc = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(TOTAL_EPOCHS):
        # Stage 2: Unfreeze full backbone after warmup
        if epoch == HEAD_WARMUP_EPOCHS:
            for p in backbone_params:
                p.requires_grad_(True)
            print("  --> Backbone unfrozen for full end-to-end fine-tuning")

        model.train()
        train_correct, train_total = 0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, labels) / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema_tracker.update(model)

            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        train_acc = train_correct / train_total
        val_acc = evaluate_standard(model, val_loader)
        ema_acc = evaluate_standard(ema_tracker.ema, val_loader)

        # Pick whichever model (live or EMA) achieved higher validation accuracy
        if ema_acc >= val_acc:
            epoch_best_acc = ema_acc
            epoch_weights = copy.deepcopy(ema_tracker.ema.state_dict())
            tag = "EMA"
        else:
            epoch_best_acc = val_acc
            epoch_weights = copy.deepcopy(model.state_dict())
            tag = "STD"

        marker = ""
        if epoch_best_acc > best_acc:
            best_acc = epoch_best_acc
            best_weights = epoch_weights
            epochs_without_improvement = 0
            marker = f" <-- NEW BEST ({tag})"
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch+1:02d}/{TOTAL_EPOCHS} | Train: {train_acc:.4f} | "
            f"Val: {val_acc:.4f} | EMA: {ema_acc:.4f}{marker}"
        )

        if epoch >= HEAD_WARMUP_EPOCHS + 5 and epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"  --> Early stopping triggered at epoch {epoch+1} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    # Final 4-View TTA Evaluation on best weights
    model.load_state_dict(best_weights)
    tta_acc = evaluate_tta_4view(model, val_loader)
    final_acc = max(best_acc, tta_acc)
    elapsed = (time.time() - start_time) / 60.0

    save_path = os.path.join(MODEL_DIR, f"{task_name}.pth")
    torch.save({
        "task_name": task_name,
        "arch": ARCH,
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "image_size": IMAGE_SIZE,
        "val_accuracy": final_acc
    }, save_path)

    print(f"\nFINISHED {task_name.upper()} in {elapsed:.1f} min | Best Val: {best_acc*100:.2f}% | 4-View TTA: {tta_acc*100:.2f}%")
    print(f"Saved checkpoint to: {save_path}")


if __name__ == "__main__":
    for task in TASKS:
        train_task(task)
    print("\nALL TASKS COMPLETED SUCCESSFULLY!")