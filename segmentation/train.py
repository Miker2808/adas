import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models.resnet_unet import RESNET18_UNET
from models.vggunet import VGG_UNET
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
NUM_WORKERS = 8
IMAGE_HEIGHT = 16*30
IMAGE_WIDTH = 16*40
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_VAL_SPLIT = 0.85
EARLY_STOPPING_PATIENCE = 10
MIN_DELTA = 0.001
DICE_BCE_ALPHA = 0.7

MODEL_PATH = "weights/residual_unet_weights.pth.tar"
SAVE_PREDICTIONS = False

# Single dataset directories
IMAGE_DIR = "dataset/images/"
MASK_DIR = "dataset/masks/"



def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.amp.autocast(DEVICE): # type: ignore
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
            # Resize
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),

            # Geometry
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent=(-0.05, 0.05),
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                p=0.5,
            ),
            A.Perspective(
                scale=(0.05, 0.1),
                p=0.3,
            ),

            # Color / lighting
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.4,
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.2,
            ),

            # Weather
            A.RandomFog(
                fog_coef_range=(0.1, 0.3),
                alpha_coef=0.1,
                p=0.2,
            ),
            A.RandomRain(
                slant_range=(-10, 10),
                drop_length=10,
                drop_width=1,
                p=0.2,
            ),

            # Noise
            A.GaussNoise(
                std_range=(0.02, 0.05),
                mean_range=(0.0, 0.0),
                per_channel=True,
                noise_scale_factor=1.0,
                p=0.5
            ),

            # Occlusion
            A.CoarseDropout(
                num_holes_range=(4, 8),
                hole_height_range=(10, 20),
                hole_width_range=(10, 20),
                p=0.2,
            ),
            
            # Normalization
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ]
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

    model = RESNET18_UNET(in_channels=3, out_channels=1).to(DEVICE)
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
        threshold=0.01,
        factor=0.2,
        patience=3,
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
    scaler = torch.amp.GradScaler('cuda') # type: ignore

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
            model.load_state_dict(early_stopping.best_model_state) # type: ignore
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