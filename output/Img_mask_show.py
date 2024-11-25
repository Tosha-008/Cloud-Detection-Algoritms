import numpy as np
import matplotlib.pyplot as plt

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

        if i == mask.shape[-1] - 2:
            min_val = np.min(mask[:, :, i])
            max_val = np.max(mask[:, :, i])
            print(f"Mask Channel {i + 1}: min = {min_val}, max = {max_val}")

    plt.show()

image = np.load('/Users/tosha_008/Downloads/Sentinel_2/subscenes_splited_384/S2B_MSIL1C_20180423T171859_N0206_R012_T14SMJ_20180423T204026_4.npy')
mask = np.load('/Users/tosha_008/Downloads/Sentinel_2/masks_splited_384/S2B_MSIL1C_20180423T171859_N0206_R012_T14SMJ_20180423T204026_4.npy')
print(mask.shape)
# dataset_path = "/Volumes/Vault/Splited_biome_384"
# test_loader_path = "/Users/mbc-air/Downloads/cloudFCN_master-Tosha-008-colab_1/cache_valid_Biome_19.pkl"
# dataset_name = "Biome"
#
# bands = [3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11]
# num_batches_to_show = 15
# patch_size=384
# num_classes = 1
# num_channels = len(bands)
#
# test_set = load_paths(test_loader_path)
#
# if not test_set:
#     train_path, valid_paths, test_paths = train_valid_test(dataset_path,
#                                                            train_ratio=0.7,
#                                                            test_ratio=0.1,
#                                                            dataset=dataset_name,
#                                                            only_test=False,
#                                                            no_test=False)
#     test_set = LandsatDataset(test_paths, cache_file=f"test_paths_{dataset_name}_{len(test_paths)}.pkl", save_cache=True)
#
# test_loader = loader.dataloader(
#     test_set, 1, patch_size,
#     transformations=[trf.train_base(patch_size, fixed=True),
#                      trf.band_select(bands),
#                      trf.class_merge(3, 4),   #  If Biome
#                      trf.class_merge(1, 2),   #  If Biome
#                      trf.normalize_to_range()
#                      ],
#     shuffle=False,
#     num_classes=num_classes,
#     num_channels=num_channels,
#     left_mask_channels=num_classes)
#
# for i, (images, masks) in enumerate(test_loader()):
#     if i >= num_batches_to_show:
#         break
#
#     print(f"Batch {i + 1}:")
#     # print(f"Images shape: {images.shape}")
#     # print(f"Masks shape: {masks.shape}")
#
#     for j in range(images.shape[0]):
#         image = images[j]
#         mask = masks[j]
#         show_image_mask(image, mask)

show_image_mask(image, mask)

