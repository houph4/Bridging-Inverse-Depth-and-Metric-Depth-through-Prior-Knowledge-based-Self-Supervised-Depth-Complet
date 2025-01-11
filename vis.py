import os
import cv2

def process_and_save_depth_maps(input_folder, output_folder):
    """
    Load all depth maps from a folder, apply JET colormap, and save them to another folder.
    :param input_folder: Path to the input folder containing depth maps
    :param output_folder: Path to the output folder for saving color depth maps
    """
    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print(f"The input folder {input_folder} does not exist!")
        return

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all image files in the folder
    file_list = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    
    if not file_list:
        print("No image files found in the input folder.")
        return

    print(f"Found {len(file_list)} image files, starting processing...")
    
    # Iterate through each depth map in the folder
    for i, file_name in enumerate(file_list):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)  # Keep the original file name

        # Read the depth map (in grayscale mode)
        depth_map = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if depth_map is None:
            print(f"Unable to read file: {file_name}, skipping...")
            continue

        # Normalize the depth map to [0, 255]
        normalized_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        normalized_depth = normalized_depth.astype('uint8')

        # Apply JET colormap
        depth_colormap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)

        # Save to the output folder
        cv2.imwrite(output_path, depth_colormap)
        print(f"Processed and saved: {output_path}")

    print("All depth maps have been processed successfully!")

# Input and output folder paths
input_folder = "output_inference_pred"  # Replace with the input folder path
output_folder = "jet_nyu_sm"  # Replace with the output folder path

# Call the function to process and save depth maps
process_and_save_depth_maps(input_folder, output_folder)

