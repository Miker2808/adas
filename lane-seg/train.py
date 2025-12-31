import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from unet import UNET
from vggunet import VGG_UNET
from dataset import TUSimpleDataset
from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
    get_val_loss,
    EarlyStopping,
    CombinedLoss
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 50
NUM_WORKERS = 4
IMAGE_HEIGHT = 16*15
IMAGE_WIDTH = 16*20
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_VAL_SPLIT = 0.85
EARLY_STOPPING_PATIENCE = 10
MIN_DELTA = 0.001
DICE_BCE_ALPHA = 0.6

MODEL_PATH = "model/vgg_unet_bn.pth.tar"
SAVE_PREDICTIONS = False

# Single dataset directories
IMAGE_DIR = "dataset/TUSimpleProcessed/images/"
MASK_DIR = "dataset/TUSimpleProcessed/masks/"



def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.amp.autocast(DEVICE):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # Unscale before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Add gradient clipping
        scaler.step(optimizer)
        scaler.update()
        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=240, width=320),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.CoarseDropout(
                num_holes_range=(4, 8),
                hole_height_range=(10, 20),
                hole_width_range=(10, 20),
                fill_value=0,
                p=0.2
            ),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            ToTensorV2()
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
                max_pixel_value=255.0,
            ),
            ToTensorV2()
        ]
    )

    # Create full dataset with train transforms (we'll handle val transforms separately)
    full_dataset = TUSimpleDataset(IMAGE_DIR, MASK_DIR, transform=train_transform)
    
    # Calculate split sizes
    dataset_size = len(full_dataset)
    train_size = int(TRAIN_VAL_SPLIT * dataset_size)
    val_size = dataset_size - train_size
    
    # Split the dataset
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Override transform for validation set
    val_dataset_with_transform = TUSimpleDataset(IMAGE_DIR, MASK_DIR, transform=val_transform)
    val_dataset.dataset = val_dataset_with_transform

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )

    model = VGG_UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = CombinedLoss(alpha=DICE_BCE_ALPHA)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=MIN_DELTA
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load(MODEL_PATH), model)

    print(f"Initial validation accuracy:")
    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # Get validation loss for scheduler and early stopping
        val_loss = get_val_loss(val_loader, model, loss_fn, DEVICE)
        print(f"Validation Loss: {val_loss:.5f}")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        check_accuracy(val_loader, model, device=DEVICE)
        
        # Check early stopping
        if early_stopping(val_loss, model):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            # Load best model
            model.load_state_dict(early_stopping.best_model_state)
            break

        # Save model checkpoint
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        }
        save_checkpoint(checkpoint, filename=MODEL_PATH)

        # Save predictions every 5 epochs
        if (epoch + 1) % 5 == 0 and SAVE_PREDICTIONS:
            save_predictions_as_imgs(
                val_loader, model, folder=f"saved_images/epoch_{epoch+1}/", device=DEVICE
            )

    # Final evaluation and save
    print("\nFinal validation accuracy:")
    check_accuracy(val_loader, model, device=DEVICE)
    if SAVE_PREDICTIONS:
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/final/", device=DEVICE
        )


if __name__ == "__main__":
    main()