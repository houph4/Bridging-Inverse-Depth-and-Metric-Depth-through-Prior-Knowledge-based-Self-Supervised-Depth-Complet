import torch
import torch.optim as optim
from loss import UcertRELossL1
from foundationdc import Foundationdc
from torch.utils.data import DataLoader, Subset
from datasets.nyu import NYU
from tqdm import tqdm
from eavl import *
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt

def set_random_seed(seed):
    """
    Set the random seed for reproducibility.
    
    Args:
        seed (int): The seed value for random number generators.
    """
    torch.manual_seed(seed)  # Set seed for CPU
    torch.cuda.manual_seed(seed)  # Set seed for GPU (if using GPU)
    torch.backends.cudnn.deterministic = True  # Ensure reproducibility
    torch.backends.cudnn.benchmark = False  # Improve reproducibility in dynamic computation graphs
    np.random.seed(seed)  # Set seed for NumPy

def load_trained_model(model_path=None, device='cuda'):
    """
    Load a trained model. If no model_path is provided, only initialize the model without loading pre-trained weights.
    
    Args:
        model_path (str, optional): Path to the pre-trained model weights.
        device (str): The device to load the model onto (CPU or GPU).
        
    Returns:
        model: The initialized model (with or without pre-trained weights).
    """
    model = Foundationdc().to(device)  # Initialize the model

    if model_path:  # Check if model_path is provided
        checkpoint = torch.load(model_path)  # Load the complete checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])  # Load only model weights
        print(f"Model loaded from {model_path}")
    else:
        print("No model_path provided. Initialized model without pre-trained weights.")

    return model

def count_parameters(model):
    """
    Print the size of parameters in each layer and the total number of parameters.
    
    Args:
        model: The neural network model.
    """
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name}, Size: {param.size()}, Params: {param.numel()}")
            total_params += param.numel()
    print(f"Total Trainable Parameters: {total_params}")

def validate_model(model, val_loader, device='cuda'):
    """
    Calculate various error metrics using the validation dataset.
    
    Args:
        model: The trained model.
        val_loader: DataLoader for validation data.
        device (str): The device to use (CPU or GPU).
        
    Returns:
        mean_errors (np.array): The mean errors of the metrics.
    """
    model.eval()  # Set model to evaluation mode
    errors = []

    with torch.no_grad():  # Disable gradient calculation
        progress_bar = tqdm(val_loader, desc="Validation Progress")
        for batch_idx, (rgb, lidar, metric_depth, real_depth, _) in enumerate(progress_bar):
            rgb = rgb.to(device)
            lidar = lidar.to(device)
            metric_depth = metric_depth.to(device)
            real_depth = real_depth.to(device)

            # Forward pass
            output, uncertain = model(lidar, metric_depth)
            predicted_depth_np = output.squeeze().cpu().numpy()
            real_depth_np = real_depth.squeeze().cpu().numpy()

            # Compute mask for valid pixels
            knn_valid_mask_np_real = (real_depth > 0.2) & (real_depth < 5.0)
            knn_valid_mask_np_real = knn_valid_mask_np_real.squeeze().cpu().numpy().astype(bool)

            # Compute errors
            metrics_with_mask = compute_errors(real_depth_np, predicted_depth_np, knn_valid_mask_np_real)
            errors.append(metrics_with_mask)

    # Calculate mean errors across all samples
    mean_errors = np.array(errors).mean(0)

    # Print results
    print("\n  " + ("{:>11} | " * 10).format("abs_rel", "sq_rel", "rmse", "logrmse", "irmse", "imae", "a1", "a2", "a3", "mae"))
    print(("&{: 11.3f}  " * 10).format(*mean_errors.tolist()) + "\\\\")

    return mean_errors

def train_model_with_loader(model, optimizer, scheduler, train_loader, val_loader, epochs=40, device='cuda', start_epoch=0):
    """
    Train the model using a DataLoader with progress bars.
    
    Args:
        model: The trained model.
        optimizer: The optimizer for training the model.
        scheduler: The learning rate scheduler.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        epochs (int): Number of training epochs.
        device (str): The device to use (CPU or GPU).
        start_epoch (int): Starting epoch for training.
    
    Returns:
        model: The trained model.
    """
    model = model.to(device)  # Move model to the correct device
    model.train()  # Set the model to training mode

    for epoch in range(start_epoch, epochs):
        total_loss = 0.0
        # Create a progress bar with tqdm
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch+1}/{epochs}]")

        for batch_idx, (rgb, lidar, metric_depth, depth, _) in progress_bar:
            # Move data to the device
            rgb = rgb.to(device)
            lidar = lidar.to(device)
            metric_depth = metric_depth.to(device)
            depth = depth.to(device)

            optimizer.zero_grad()  # Reset gradients

            output, uncertain = model(lidar, metric_depth)

            # Compute the loss
            loss_fn = UcertRELossL1()
            loss = loss_fn(output, uncertain, lidar, metric_depth)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            # Update total loss
            total_loss += loss.item()

            # Update tqdm progress bar with loss value
            progress_bar.set_postfix({"Batch Loss": loss.item()})

        # Step the scheduler
        scheduler.step()

        # Print epoch stats
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.4f}")

        # Run validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Running Validation after Epoch {epoch + 1}")
            validate_model(model, val_loader, device)

        # Optional: Save the model at intermediate epochs
        if (epoch + 1) % 1 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, f'model_epoch_nyv_{epoch+1}.pt')

    # Save the final model
    torch.save({
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, 'model_final.pt')

    return model

# Training loop and data preparation
if __name__ == "__main__":
    # Set random seed for reproducibility
    set_random_seed(17)
    subset_ratio = 0.2  # Use 20% of training data

    # Prepare training dataset
    train_dataset = NYU(
        path='datas/nyudepthv2',
        mode='train',  # Training mode
        height=256,  # Image height
        width=320,  # Image width
        RandCrop=True  # Enable random cropping
    )

    # Create a subset of the training dataset
    subset_size = int(len(train_dataset) * subset_ratio)
    indices = np.random.permutation(len(train_dataset))[:subset_size]  # Randomly select sample indices
    train_subset = Subset(train_dataset, indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Prepare validation dataset
    val_dataset = NYU(
        path='datas/nyudepthv2',
        mode='val',
        height=256,
        width=320,
        RandCrop=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Load model, optimizer, and scheduler
    model = load_trained_model()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=80, gamma=0.1)

    # Train the model
    trained_model = train_model_with_loader(model, optimizer, scheduler, train_loader, val_loader, epochs=100)

