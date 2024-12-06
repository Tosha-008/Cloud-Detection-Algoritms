import numpy as np
import matplotlib.pyplot as plt
from Fmask.compare_fmask_mask_defs import sentinel_13_to_11
from data.Datasets import train_valid_test, LandsatDataset
from data import loader, transformations as trf
from data.loader import load_paths


def normalize_to_255(image):
    """
    Normalizes the values of the array to the range from 0 to 255.

    Parameters:
    image (numpy array): The input image (array) to be normalized.

    Returns:
    numpy array: The normalized image with a range of [0, 255].
    """

    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image =  255 * (image - min_val) / (max_val - min_val)
    return normalized_image.astype(np.uint8)


def show_image_mask(image, mask):
    fig, axes = plt.subplots(1, 1 + mask.shape[-1], figsize=(15, 5))

    image_rgb = image[:, :, :3]
    axes[0].imshow(image_rgb)
    axes[0].set_title("Image")
    axes[0].axis("off")

    for i in range(mask.shape[-1]):
        axes[i + 1].imshow(mask[:, :, i], cmap="gray", vmin=0, vmax=1)
        axes[i + 1].set_title(f"Mask Channel {i + 1}")
        axes[i + 1].axis("off")

        if i == mask.shape[-1] - 2:
            min_val = np.min(mask[:, :, i])
            max_val = np.max(mask[:, :, i])
            print(f"Mask Channel {i + 1}: min = {min_val}, max = {max_val}")

    plt.show()

image_sent = np.load('/media/ladmin/Vault/Sentinel_2/subscenes/S2A_MSIL1C_20180416T081601_N0206_R121_T34HDH_20180416T115825.npy')
image = np.load('/media/ladmin/Vault/Splited_set_2_384/LC08_L1GT_117036_20150711_20200908_02_T2/005011/image.npy')
# mask = np.load('/Users/tosha_008/Downloads/Sentinel_2/masks_splited_384/S2B_MSIL1C_20180423T171859_N0206_R012_T14SMJ_20180423T204026_4.npy')


# image = sentinel_13_to_11(image_sent, variant=2)
image = normalize_to_255(image)
num_channels = image.shape[-1]

# if num_channels != 13:
#     print(f"Warning: Expected 12 channels but found {num_channels}")

plt.figure(figsize=(15, 15))
for i in range(num_channels):
    plt.subplot(4, 4, i + 1)
    plt.imshow(image[:, :, i], cmap='gray')
    plt.title(f"Channel {i + 1}")
    plt.axis('off')

plt.tight_layout()
plt.show()


