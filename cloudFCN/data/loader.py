import numpy as np
import os

from matplotlib import pyplot as plt
from scipy import misc
import time


def dataloader(dataset, batch_size, patch_size, transformations=None,
               shuffle=False, num_classes=5, num_channels=12):
    """
    Function to create pipeline for dataloading, creating batches of image/mask pairs.

    Parameters
    ----------
    dataset : Dataset
        Contains paths for all image/mask pairs to be sampled.
    batch_size : int
        Number of image/mask pairs per batch.
    patch_size : int
        Spatial dimension of each image/mask pair (assumes width==height).
    transformations : list, optional
        Set of transformations to be applied in series to image/mask pairs
    shuffle : bool, optional
        True for randomized indexing of dataset, False for ordered.
    num_classes : int, optional
        Number of classes in masks.
    num_channels : int, optional
        Number of channels in images.

    Returns
    -------
    generator : iterator
        Yields batches of image/mask pairs.
    """

    if batch_size > len(dataset) and not shuffle:
        raise ValueError('Dataset is too small for given batch size.')

    def generator(batch_size=batch_size):
        offset = 0

        while True:
            if not shuffle:
                idxs = list(range(offset, offset + batch_size))
                offset = (offset + batch_size) % (len(dataset) - batch_size - 1)
            elif len(dataset) >= batch_size:
                idxs = np.random.choice(len(dataset), batch_size, replace=False)
            else:
                idxs = np.random.choice(len(dataset), batch_size, replace=True)

            ims = np.empty([batch_size, patch_size, patch_size, num_channels])
            masks = np.empty([batch_size, patch_size, patch_size, num_classes])

            for batch_i, idx in enumerate(idxs):
                im, mask = dataset[idx]

                if transformations:
                    for transform in transformations:
                        im, mask = transform(im, mask)

                if np.any(np.isnan(im)) or np.any(np.isinf(im)):
                    print(f"Invalid image data at index {idx}: {im}")
                if np.any(np.isnan(mask)) or np.any(np.isinf(mask)):
                    print(f"Invalid mask data at index {idx}: {mask}")

                ims[batch_i] = im
                masks[batch_i] = mask

            yield ims, masks

    return generator


# def get_min_max(loader_function, foga=False):
#     """Calculate min and max values from the first batch of the data loader and display an RGB image with its mask."""
#     first_batch = next(iter(loader_function()))
#     images, masks = first_batch
#
#     min_val = np.min(images)
#     max_val = np.max(images)
#
#     min_mask = np.min(masks)
#     max_mask = np.max(masks)
#
#     plt.figure(figsize=(10, 5))
#
#     plt.subplot(1, 2, 1)
#     rgb_image = images[0, :, :, :3]
#     plt.imshow(rgb_image)
#     plt.title('RGB Image')
#     plt.axis('off')
#
#     plt.subplot(1, 2, 2)
#     plt.imshow(masks[0], cmap='gray')
#     plt.title('Mask')
#     plt.axis('off')
#
#     plt.show()
#
#     if foga:
#         return min_val, max_val, min_mask, max_mask, images, masks
#
#     else:
#         return min_val, max_val, min_mask, max_mask




if __name__ == "__main__":
    from Datasets import LandsatDataset
    import transformations as trf
    import sys

    patch_size = 398
    batch_size = 5
    train = LandsatDataset(sys.argv[1])
    print(f"Total images in dataset: {len(train.paths)}")

    transformations = [trf.train_base(patch_size)]
    train_ = dataloader(train, batch_size, patch_size, num_channels=12, num_classes=5,
                        transformations=transformations, shuffle=True)
    train_flow = train_()

    for i in range(30):
        fig, ax = plt.subplots(2, 5, figsize=(10, 4))
        t1 = time.time()
        imgs, masks = next(train_flow)
        t2 = time.time()

        for j in range(5):
            imrgb, maskflat = train.display(imgs[j, ...], masks[j, ...])
            ax[0, j].imshow(imrgb)
            ax[0, j].set_axis_off()
            ax[1, j].imshow(maskflat, cmap='viridis')
            ax[1, j].set_axis_off()
        plt.show()
