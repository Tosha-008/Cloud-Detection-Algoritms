import numpy as np
import os
import random
from matplotlib import pyplot as plt
from scipy import misc
import time
import pickle
import yaml


def dataloader(dataset, batch_size, patch_size, transformations=None,
               shuffle=False, num_classes=5, num_channels=12, left_mask_channels=3):
    """
    Function to create pipeline for dataloading, creating batches of image/mask pairs.

    Parameters
    ----------
    left_mask_chanels
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

                if isinstance(im, str) or isinstance(mask, str):
                    im = np.load(im, allow_pickle=True).astype('float')
                    mask = np.load(mask, allow_pickle=True)

                if transformations:
                    for transform in transformations:
                        im, mask = transform(im, mask)

                if np.any(np.isnan(im)) or np.any(np.isinf(im)):
                    print(f"Invalid image data at index {idx}: {im}")
                if np.any(np.isnan(mask)) or np.any(np.isinf(mask)):
                    print(f"Invalid mask data at index {idx}: {mask}")

                if left_mask_channels:
                    mask = mask[:, :, -left_mask_channels:]
                ims[batch_i] = im
                masks[batch_i] = mask

            yield ims, masks

    return generator


def dataloader_descriptors(dataset, batch_size, transformations=None,
                           shuffle=False, descriptor_list=None, left_mask_channels=3,
                           band_policy=None):
    """
    Function to create a pipeline for dataloading, creating batches of image/mask pairs,
    with the ability to process descriptors.

    Parameters
    ----------
    ...
    band_policy : str / list / int / tuple, optional
        Policy for selecting bands. If tuple, a fixed policy will be generated for each batch.

    Returns
    -------
    generator : iterator
        Yields batches of (images, masks, descriptors).
    """
    if batch_size > len(dataset) and not shuffle:
        raise ValueError("Dataset is too small for the given batch size.")

    def generator(batch_size=batch_size):
        offset = 0

        while True:
            # Determine band_policy for this batch
            fixed_band_policy = None
            if isinstance(band_policy, tuple):
                fixed_band_policy = random.randint(band_policy[0], band_policy[1])
            else:
                fixed_band_policy = band_policy

            if not shuffle:
                idxs = list(range(offset, offset + batch_size))
                offset = (offset + batch_size) % (len(dataset) - batch_size - 1)
            elif len(dataset) >= batch_size:
                idxs = np.random.choice(len(dataset), batch_size, replace=False)
            else:
                idxs = np.random.choice(len(dataset), batch_size, replace=True)

            # Initialize batch arrays
            ims = []
            masks = []
            descriptors_array = []

            for batch_i, idx in enumerate(idxs):
                im, mask = dataset[idx]

                if isinstance(im, str) or isinstance(mask, str):
                    im = np.load(im, allow_pickle=True).astype("float")
                    im = im[..., :-1]
                    mask = np.load(mask, allow_pickle=True)

                descriptors = descriptor_list  # Default descriptors
                if transformations:
                    for transform in transformations:
                        im, mask, descriptors = transform(im, mask, descriptors, band_policy=fixed_band_policy)

                im = np.moveaxis(im, -1, 0)[..., np.newaxis]

                if np.any(np.isnan(im)) or np.any(np.isinf(im)):
                    print(f"Invalid image data at index {idx}: {im}")
                if np.any(np.isnan(mask)) or np.any(np.isinf(mask)):
                    print(f"Invalid mask data at index {idx}: {mask}")

                # Add data to the batch
                if left_mask_channels:
                    mask = mask[:, :, -left_mask_channels:]

                ims.append(im)
                masks.append(mask)
                descriptors_array.append(descriptors)

            ims = np.array(ims, dtype=np.float32)
            masks = np.array(masks, dtype=np.float32)
            descriptors_array = np.array(descriptors_array, dtype=np.float32)

            yield [(ims, descriptors_array), masks]

    return generator




def combined_generator(sentinel_gen, landsat_gen, sentinel_weight=0.3, landsat_weight=0.7, seed=None):
    if seed is None:
        seed = random.randint(0, 2**32 -1)

    random.seed(seed)
    np.random.seed(seed)
    sentinel_prob = sentinel_weight / (sentinel_weight + landsat_weight)

    while True:
        if np.random.rand() < sentinel_prob:
            yield next(sentinel_gen)
        else:
            yield next(landsat_gen)


def convert_paths_to_tuples(paths_list):
    return [(os.path.join(path, 'image.npy'), os.path.join(path, 'mask.npy')) for path in paths_list]


def load_paths(filename="dataloader.pkl", valid=False):
    if filename is None:
        print("No existing datapaths found. Creating a new one.")
        if valid:
            return None, None
        else:
            return None
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            datapaths = pickle.load(f)
        print("Datapaths loaded from", filename)
        datapaths_copy = datapaths[:]
        random.shuffle(datapaths_copy)
        if valid:
            return convert_paths_to_tuples(datapaths_copy), datapaths
        else:
            return convert_paths_to_tuples(datapaths_copy)
    else:
        print("No existing datapaths found. Creating a new one.")
        return None


import yaml

def generate_descriptor_list_from_yaml(yaml_path):
    """
    Creates a descriptor list based on a YAML metadata file.

    Parameters:
    -----------
    yaml_path : str
        Path to the YAML metadata file.

    Returns:
    --------
    descriptor_list : list
        List where each entry is a descriptor for a channel (a list of three values:
        [left wavelength, center wavelength, right wavelength]).
    """
    try:
        # Load the YAML file
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)

        # Check for the presence of the 'bands' key
        if 'bands' not in metadata:
            raise ValueError("YAML file must contain the 'bands' key.")

        # Initialize the descriptor list
        descriptor_list = []

        # Generate descriptors for each channel
        for band, band_info in metadata['bands'].items():
            # Extract the center wavelength and band width
            centre = band_info.get('band_centre')
            width = band_info.get('band_width')

            if centre is None or width is None:
                raise ValueError(f"Channel {band} is missing 'band_centre' or 'band_width' information.")

            # Calculate the left and right wavelengths
            left = centre - (width / 2)
            right = centre + (width / 2)

            # Add the descriptor to the list
            descriptor_list.append([left, centre, right])

        return descriptor_list

    except FileNotFoundError:
        print(f"The file {yaml_path} was not found.")
        return []
    except yaml.YAMLError as e:
        print(f"Error reading the YAML file: {e}")
        return []
    except ValueError as e:
        print(f"Error in the YAML file structure: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []


if __name__ == "__main__":
    print("Loading paths...")
