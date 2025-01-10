def replace_image_with_pred_depth(input_file, output_file):
    """
    Read the input_file, replace 'image' in each line with 'pred_depth', and write to output_file.
    """
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Replace 'image' with 'pred_depth'
    modified_lines = [line.replace('/image/', '/pred_depth/') for line in lines]

    # Write the modified content to a new file
    with open(output_file, 'w') as file:
        file.writelines(modified_lines)

if __name__ == "__main__":
    input_file = 'val_image.txt'  # Input file ##Replace it with 'train_image.txt','test_image.txt','val_image.txt'
    output_file = 'val_pred.txt'  # Output new file ##Replace it with 'train_pred.txt','test_pred.txt','val_pred.txt'
    replace_image_with_pred_depth(input_file, output_file)
    print(f"Successfully replaced 'image' with 'pred_depth' in {input_file} and wrote to {output_file}.")

