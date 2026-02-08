import torch
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import _LRScheduler # Added for warmup
from models.resnet_unet import RESNET18_UNET
from models.vggunet import VGG_UNET
from dataset import TUSimpleDataset
from loss_function import LaneSegmentationLoss
from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
    get_val_loss,
    EarlyStopping
)

# Hyperparameters.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 50
WARMUP_EPOCHS = 5
NUM_WORKERS = 8
IMAGE_HEIGHT = 16*30
IMAGE_WIDTH = 16*40
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_VAL_SPLIT = 0.85
LEARNING_RATE = 1e-4
ES_PATIENCE = 5
ES_MIN_DELTA = 0.001
REDUCE_LR_FACTOR = 0.2
REDUCE_LR_MIN_DELTA = 5e-4
REDUCE_LR_MIN_LR = 1e-6
REDUCE_LR_THRESHOLD = 0.1
REDUCE_LR_PATIENCE = 2
WEIGHT_DECAY = 5e-4
BCE_WEIGHT = 0.5
TVERSKY_ALPHA = 0.7

MODEL_PATH = "weights/residual_unet_weights.pth.tar"
SAVE_PREDICTIONS = True

# Single dataset directories
IMAGE_DIR = "dataset/images/"
MASK_DIR = "dataset/masks/"

class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, target_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = (self.last_epoch + 1) / self.warmup_epochs * self.target_lr
            return [lr for _ in self.base_lrs]
        return [group['lr'] for group in self.optimizer.param_groups]

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        with torch.amp.autocast(DEVICE): # type: ignore
            predictions = model(data)

            if batch_idx % 50 == 0:
                probs = torch.sigmoid(predictions)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=(-0.05, 0.05), scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
            A.RandomShadow(num_shadows_limit=(1, 3), shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=0.3),
            A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.1, p=0.2),
            A.RandomRain(slant_range=(-10, 10), drop_length=10, drop_width=1, p=0.2),
            A.GaussNoise(std_range=(0.02, 0.05), mean_range=(0.0, 0.0), per_channel=True, p=0.5),
            A.CoarseDropout(num_holes_range=(4, 8), hole_height_range=(10, 20), hole_width_range=(10, 20), p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ] # type: ignore
    )

    val_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ]
)

    full_dataset = TUSimpleDataset(IMAGE_DIR, MASK_DIR)
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(TRAIN_VAL_SPLIT * dataset_size)
    
    random.seed(42)
    random.shuffle(indices)
    
    train_indices, val_indices = indices[:split], indices[split:]

    # 2. Create separate datasets with their own transforms
    train_dataset = TUSimpleDataset(IMAGE_DIR, MASK_DIR, transform=train_transform)
    val_dataset = TUSimpleDataset(IMAGE_DIR, MASK_DIR, transform=val_transform)

    # 3. Create Subsets using the indices
    train_loader = DataLoader(
        torch.utils.data.Subset(train_dataset, train_indices), 
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(val_dataset, val_indices), 
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False
    )

    model = RESNET18_UNET(in_channels=3, out_channels=1).to(DEVICE)
    criterion = LaneSegmentationLoss(bce_weight=BCE_WEIGHT, tversky_alpha=TVERSKY_ALPHA)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))

    # Initialize schedulers
    warmup_scheduler = LinearWarmupScheduler(optimizer, WARMUP_EPOCHS, LEARNING_RATE)
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        threshold=REDUCE_LR_THRESHOLD,
        factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE,
        min_lr=REDUCE_LR_MIN_LR,
    )

    early_stopping = EarlyStopping(patience=ES_PATIENCE, min_delta=ES_MIN_DELTA)

    if LOAD_MODEL:
        load_checkpoint(MODEL_PATH, model)

    print(f"Initial validation accuracy:")
    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.amp.GradScaler('cuda') # type: ignore

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Display current LR for monitoring
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.2e}")

        train_fn(train_loader, model, optimizer, criterion, scaler)
        val_loss = get_val_loss(val_loader, model, criterion, DEVICE)
        print(f"Validation Loss: {val_loss:.5f}")
        
        # Scheduler Step Logic
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(val_loss)
        
        check_accuracy(val_loader, model, device=DEVICE)
        
        if early_stopping(val_loss, model):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            model.load_state_dict(early_stopping.best_model_state) # type: ignore
            break

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        }
        save_checkpoint(checkpoint, filename=MODEL_PATH)

        if (epoch + 1) % 5 == 0 and SAVE_PREDICTIONS:
            save_predictions_as_imgs(val_loader, model, folder=f"saved_images/epoch_{epoch+1}/", device=DEVICE)
        
    print("\nFinal validation accuracy:")
    check_accuracy(val_loader, model, device=DEVICE)
    if SAVE_PREDICTIONS:
        save_predictions_as_imgs(val_loader, model, folder="saved_images/final/", device=DEVICE)

if __name__ == "__main__":
    main()