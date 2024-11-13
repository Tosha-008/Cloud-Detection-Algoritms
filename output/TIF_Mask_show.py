import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os

# Paths to your TIFF files
tiff_paths = [
    '/Volumes/Vault/Row_set_2/LC08_L1TP_006048_20191101_20200825_02_T1/LC08_L1TP_006048_20191101_20200825_02_T1_B4.TIF',   # Red channel
    '/Volumes/Vault/Row_set_2/LC08_L1TP_006048_20191101_20200825_02_T1/LC08_L1TP_006048_20191101_20200825_02_T1_B3.TIF', # Green channel
    '/Volumes/Vault/Row_set_2/LC08_L1TP_006048_20191101_20200825_02_T1/LC08_L1TP_006048_20191101_20200825_02_T1_B2.TIF'   # Blue channel
]

# Path to the mask file
image_path = '/Volumes/Vault/Row_set_2/LC08_L1TP_006048_20191101_20200825_02_T1/LC08_L1TP_006048_20191101_20200825_02_T2_fixedmask.TIF'

# Check if TIFF files exist
for path in tiff_paths:
    if not os.path.exists(path):
        print(f"File not found: {path}")

# Load images and combine them into RGB
rgb_image = None
try:
    red_band = rasterio.open(tiff_paths[0]).read(1)   # Read the red channel
    green_band = rasterio.open(tiff_paths[1]).read(1) # Read the green channel
    blue_band = rasterio.open(tiff_paths[2]).read(1)  # Read the blue channel

    # Combine channels into a single RGB image
    rgb_image = np.stack((red_band, green_band, blue_band), axis=-1)
    print(f"RGB Image shape: {rgb_image.shape}")

    # Normalize values
    rgb_image = rgb_image.astype(np.float32)  # Convert to float for normalization
    rgb_image /= rgb_image.max()  # Normalize to the range [0, 1]

except Exception as e:
    print(f"Error loading TIFF files: {e}")

# Check if the mask file exists
if not os.path.exists(image_path):
    print(f"File not found: {image_path}")
else:
    # Load the image using rasterio
    with rasterio.open(image_path) as src:
        # Load all layers
        mask_data = src.read()  # Read all channels
        print(f"Mask Data shape: {mask_data.shape}")  # Print the shape of the data

# Display images
plt.figure(figsize=(15, 5))

# Display the RGB image
if rgb_image is not None:
    plt.subplot(1, 2, 1)  # Subplot for RGB
    plt.imshow(rgb_image)
    plt.title("RGB Image")
    plt.axis("off")
else:
    print("Not enough channels to form RGB image.")

# Display the mask
plt.subplot(1, 2, 2)  # Subplot for mask
if mask_data is not None:
    # Use cmap='gray' for displaying the mask with one channel
    plt.imshow(mask_data[0, :, :], cmap='gray')  # Since we have only one channel
    plt.title("Mask")
    plt.axis("off")
else:
    print("Failed to load mask.")

plt.show()
