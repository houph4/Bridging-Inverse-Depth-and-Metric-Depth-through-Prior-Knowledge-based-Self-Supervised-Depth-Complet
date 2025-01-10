import torch
from datasets.nyu import NYU
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
from pathlib import Path
from foundationdc import Foundationdc
import numpy as np
from eavl import *

def load_trained_model(model_path, device='cuda'):
    """
    Load the trained model.
    """
    model = Foundationdc().to(device)  # Initialize the model
    checkpoint = torch.load(model_path)  # Load the complete checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])  # Load only the model weights
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {model_path}")
    return model

def infer(model, data_loader, output_pred_dir, output_real_dir, output_metric_dir, output_sp_dir, device='cuda'):
    """
    Inference function: Use the trained model for inference.
    
    Args:
        model: The trained model.
        data_loader: Data loader providing inference data.
        output_pred_dir: Directory to save predicted depth results.
        output_real_dir: Directory to save real depth results.
        output_metric_dir: Directory to save metric depth results.
        output_sp_dir: Directory to save sparsely sampled depth results.
        device: The device to use (CPU or GPU).
    """
    model.eval()  # Set model to evaluation mode
    output_pred_dir.mkdir(parents=True, exist_ok=True)  # Create output directory
    output_real_dir.mkdir(parents=True, exist_ok=True)  # Create output real depth directory
    output_metric_dir.mkdir(parents=True, exist_ok=True)  # Create output metric depth directory
    output_sp_dir.mkdir(parents=True, exist_ok=True)  # Create output sparsely sampled depth directory
    errors = []

    with torch.no_grad():  # Disable gradient calculation
        progress_bar = tqdm(data_loader, desc="Inference Progress")
        for idx, (rgb, lidar, metric_depth, depth,_) in enumerate(progress_bar):
            # Move data to device
            rgb = rgb.to(device)
            lidar = lidar.to(device)
            metric_depth = metric_depth.to(device)
            depth = depth.to(device)

            # Forward inference
            output, uncertainty = model(lidar, metric_depth)
            predicted_depth_np = output.squeeze().cpu().numpy()
            real_depth_np = depth.squeeze().cpu().numpy()
            knn_valid_mask_np_real = (depth > 0.1) & (depth < 5) & (output > 0)
            knn_valid_mask_np_real = knn_valid_mask_np_real.squeeze().cpu().numpy().astype(bool)

            # Compute errors
            metrics_with_mask = compute_errors(real_depth_np, predicted_depth_np, knn_valid_mask_np_real)
            errors.append(metrics_with_mask)

            # Handle and save results
            for i in range(rgb.size(0)):  # Iterate through each image in the batch
                # Save predicted depth map
                pred_depth_np = output[i].squeeze().cpu().numpy()
                pred_depth_path = output_pred_dir / f"pred_depth_{idx}_{i}.png"
                cv2.imwrite(str(pred_depth_path), (pred_depth_np / pred_depth_np.max() * 255).astype(np.uint8))

                # Save real depth map
                real_depth_np = depth[i].squeeze().cpu().numpy()
                real_depth_path = output_real_dir / f"real_depth_{idx}_{i}.png"
                cv2.imwrite(str(real_depth_path), (real_depth_np / real_depth_np.max() * 255).astype(np.uint8))

                # Save metric depth map
                metric_depth_np = metric_depth[i].squeeze().cpu().numpy()
                metric_depth_path = output_metric_dir / f"metric_depth_{idx}_{i}.png"
                cv2.imwrite(str(metric_depth_path), (metric_depth_np / metric_depth_np.max() * 255).astype(np.uint8))

                # Save sparsely sampled depth map
                sp_depth_np = lidar[i].squeeze().cpu().numpy()
                sp_depth_path = output_sp_dir / f"sp_depth_{idx}_{i}.png"
                cv2.imwrite(str(sp_depth_path), (sp_depth_np / sp_depth_np.max() * 255).astype(np.uint8))

                progress_bar.set_postfix({"Processed": f"{idx * rgb.size(0) + i + 1}/{len(data_loader.dataset)}"})

    mean_errors = np.array(errors).mean(0)

    # Print results
    print("\n  " + ("{:>11} | " * 10).format("abs_rel", "sq_rel", "rmse", "logrmse", "irmse", "imae", "a1", "a2", "a3", "mae"))
    print(("&{: 11.3f}  " * 10).format(*mean_errors.tolist()) + "\\\\")

if __name__ == "__main__":
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = 'void(99-50).pt'
    output_dir = Path('output_inference_pred')
    real_dir = Path('output_inference_real')
    metr_dir = Path('output_inference_metric')
    sp_dir = Path('output_inference_sp')

    # Initialize dataset and DataLoader
    test_dataset = NYU(
        path='datas/nyudepthv2',
        mode='val',
        height=256,
        width=320,
        RandCrop=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=2,  
        shuffle=False,  
        num_workers=4,  
        pin_memory=True  
    )

    # Load model
    model = load_trained_model(model_path, device)

    # Run inference
    infer(model, test_loader, output_dir, real_dir, metr_dir, sp_dir, device)

