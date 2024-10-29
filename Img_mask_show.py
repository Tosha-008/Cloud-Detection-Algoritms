import numpy as np
import json
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
import tensorflow as tf
from cloudFCN.data.Datasets import train_valid_test, LandsatDataset
from cloudFCN.data import loader, transformations as trf
from MFCNN.model_mfcnn_def import *
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
    normalized_image = 255 * (image - min_val) / (max_val - min_val)
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

        if i == mask.shape[-1] - 1:
            min_val = np.min(mask[:, :, i])
            max_val = np.max(mask[:, :, i])
            print(f"Mask Channel {i + 1}: min = {min_val}, max = {max_val}")

    plt.show()

# image = np.load('/Volumes/Vault/Splited_data/Grass:Crops/LC82020522013141LGN01/005011/image.npy')
# mask = np.load('/Volumes/Vault/Splited_data/Grass:Crops/LC82020522013141LGN01/005011/mask.npy')
dataset_path = "/Volumes/Vault/Splited_data_set_2"

bands = [3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11]
num_batches_to_show = 15

test_paths = train_valid_test(dataset_path, dataset='Landsat_2', only_test=True)

test_set = LandsatDataset(test_paths)
test_set.randomly_reduce(0.01)

test_loader = loader.dataloader(
    test_set, 1, 398,
    transformations=[trf.train_base(398, fixed=True),
                     trf.band_select(bands),
                     trf.class_merge(2, 3),
                     # trf.class_merge(1, 2),
                     trf.normalize_to_range()
                     ],
    shuffle=False,
    num_classes=1,
    num_channels=len(bands),
    remove_mask_chanels=True)

for i, (images, masks) in enumerate(test_loader()):
    if i >= num_batches_to_show:
        break

    print(f"Batch {i + 1}:")
    # print(f"Images shape: {images.shape}")
    # print(f"Masks shape: {masks.shape}")

    for j in range(images.shape[0]):
        image = images[j]
        mask = masks[j]
        show_image_mask(image, mask)