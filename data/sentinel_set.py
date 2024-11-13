import os
import numpy as np


def split_and_save_patches(input_folder, patch_size, output_folder=None):
    if output_folder is None:
        output_folder = os.path.join(input_folder, "patches")
    os.makedirs(output_folder, exist_ok=True)

    all_files = [f for f in os.listdir(input_folder) if f.endswith(".npy")]
    total_images = len(all_files)

    if total_images == 0:
        print("No .npy files found in the input folder.")
        return

    total_patches = 0

    print(f"Starting processing. Total images: {total_images}")

    for current_image, filename in enumerate(all_files, start=1):
        image_path = os.path.join(input_folder, filename)
        image = np.load(image_path)
        height, width, channels = image.shape

        print(f"Processing image {current_image}/{total_images}: {filename}")

        if height < patch_size or width < patch_size:
            print(f"Image {filename} is smaller than the specified patch size, skipping it.")
            continue

        patch_count = 1

        for i in range(0, height - patch_size + 1, patch_size):
            for j in range(0, width - patch_size + 1, patch_size):
                if i + patch_size <= height and j + patch_size <= width:
                    patch = image[i:i + patch_size, j:j + patch_size, :]

                    patch_filename = f"{os.path.splitext(filename)[0]}_{patch_count}.npy"
                    patch_path = os.path.join(output_folder, patch_filename)

                    np.save(patch_path, patch)
                    patch_count += 1
                    total_patches += 1

        # print(f"Image {filename} has been split into {patch_count - 1} patches.")

    print(f"Processing completed. Total original images: {total_images}")
    print(f"Total patches created: {total_patches}")


input_folder = "/Users/tosha_008/Downloads/Sentinel_2/masks"
patch_size = 384
output_folder = f"{input_folder}_splited_{patch_size}"

split_and_save_patches(input_folder, patch_size, output_folder)
