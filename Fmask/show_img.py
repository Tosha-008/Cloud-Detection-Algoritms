import os
import numpy as np
import matplotlib.pyplot as plt

def show_images_from_folders(base_dir, num_images=5):
    """
    Displays a specified number of images from subfolders in the given directory.

    Parameters:
    ----------
    base_dir : str
        Path to the directory containing subfolders with images.
    num_images : int
        Number of images to display.
    """
    subfolders = [os.path.join(base_dir, folder) for folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, folder))]

    images_shown = 0

    for folder in subfolders:
        image_path = os.path.join(folder, 'image.npy')
        if os.path.exists(image_path):
            image = np.load(image_path)

            # Output min and max values of the image
            print(f"Image from {folder} - Min: {image.min()}, Max: {image.max()}")

            # Normalize image to the range [0, 1] if necessary
            image_normalized = (image - image.min()) / (image.max() - image.min()) if image.max() != image.min() else image

            plt.figure(figsize=(10, 10))
            plt.imshow(image_normalized, cmap='gray')
            plt.title(f"Image from: {folder}")
            plt.axis("off")
            plt.show()

            images_shown += 1

            if images_shown >= num_images:
                break

    if images_shown == 0:
        print("No images found for display.")
    elif images_shown < num_images:
        print(f"Only {images_shown} images were displayed, as there are no more available.")

base_directory = '/Users/tosha_008/Downloads/Fmask_masks/Splited/LC81570452014213LGN00'
show_images_from_folders(base_directory, num_images=15)
