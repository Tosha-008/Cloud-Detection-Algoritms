from osgeo import gdal
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.transform import resize

# Paths to TIFF files for RGB
tiff_paths = [
    '/Users/tosha_008/Downloads/One_picture/Barren/LC81570452014213LGN00/LC81570452014213LGN00_B4.TIF',  # Red channel
    '/Users/tosha_008/Downloads/One_picture/Barren/LC81570452014213LGN00/LC81570452014213LGN00_B3.TIF',  # Green channel
    '/Users/tosha_008/Downloads/One_picture/Barren/LC81570452014213LGN00/LC81570452014213LGN00_B2.TIF'   # Blue channel
]

# Paths to Fmask and actual mask
fmask_path = '/Users/tosha_008/Downloads/Fmask_masks/Unsplited/LC81570452014213LGN00_fmask.img'
real_mask_path = '/Users/tosha_008/Downloads/One_picture/Barren/LC81570452014213LGN00/LC81570452014213LGN00_fixedmask.img'

# Function to check if files exist
def check_files_exist(paths):
    for path in paths:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return False
    return True

# Function to load and resize an image to the target size
def load_and_resize_image(path, target_shape):
    with rasterio.open(path) as src:
        image = src.read(1)
        if image.shape != target_shape:
            image = resize(image, target_shape, mode='reflect', anti_aliasing=True)
        return image

# Main block
if check_files_exist(tiff_paths + [fmask_path, real_mask_path]):
    # Define the target shape based on the first image
    with rasterio.open(tiff_paths[0]) as src:
        target_shape = src.read(1).shape

    # Load RGB channels
    red_band = load_and_resize_image(tiff_paths[0], target_shape)
    green_band = load_and_resize_image(tiff_paths[1], target_shape)
    blue_band = load_and_resize_image(tiff_paths[2], target_shape)

    # Combine into RGB
    rgb_image = np.stack((red_band, green_band, blue_band), axis=-1)
    rgb_image = rgb_image / np.max(rgb_image)  # Normalize to the range [0, 1]

    # Load Fmask
    fmask_dataset = gdal.Open(fmask_path)
    fmask_data = fmask_dataset.GetRasterBand(1).ReadAsArray()
    if fmask_data.shape != target_shape:
        fmask_data = resize(fmask_data, target_shape, mode='reflect', anti_aliasing=True)

    # Load actual mask
    with rasterio.open(real_mask_path) as src:
        real_mask = src.read(1)
        if real_mask.shape != target_shape:
            real_mask = resize(real_mask, target_shape, mode='reflect', anti_aliasing=True)

    # Display images
    plt.figure(figsize=(15, 5))

    # RGB image
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_image)
    plt.title("RGB Image")
    plt.axis("off")

    # Actual mask
    plt.subplot(1, 3, 2)
    plt.imshow(real_mask, cmap='gray')
    plt.title("Actual Mask")
    plt.axis("off")

    # Fmask
    plt.subplot(1, 3, 3)
    plt.imshow(fmask_data, cmap='gray')
    plt.title("Fmask Mask")
    plt.colorbar()
    plt.axis("off")

    plt.show()
else:
    print("Some files are missing.")
