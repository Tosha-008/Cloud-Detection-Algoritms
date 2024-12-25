import numpy as np
import os
from random import shuffle
import math
import pickle
from random import shuffle


class Dataset():
    """
    A class used to load image/mask pairs from a set of directories

    Attributes
    ----------
    dirs : str, list
        Paths for each parent directory
    paths : list
        Paths to every subdirectory within self.dirs that contain valid image/mask pairs.
    """

    def __init__(self, dirs, orderShuffle=True, cache_file="paths_cache.pkl", save_cache=True):
        self.dirs = dirs
        self.cache_file = cache_file
        self.paths = self.load_or_parse_paths(save_cache)
        if orderShuffle:
            shuffle(self.paths)
        print(f"LandsatDataset with {len(self.paths)} tiles")

    def load_or_parse_paths(self, save_cache):
        if os.path.exists(self.cache_file):
            print("Loading paths from cache...")
            with open(self.cache_file, "rb") as f:
                paths = pickle.load(f)
        else:
            print("Parsing directories...")
            paths = self.parse_dirs()
            if save_cache:
                with open(self.cache_file, "wb") as f:
                    pickle.dump(paths, f)
                print("Paths cached for future use.")
            else:
                print("Caching skipped.")
        return paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        im = np.load(os.path.join(self.paths[index], 'image.npy')).astype('float')
        mask = np.load(os.path.join(self.paths[index], 'mask.npy'))

        try:
            return im, mask
        except BaseException:
            print("Could not read:", self.paths[index])

    def parse_dirs(self):
        """
        Look for all valid subdirectories in self.dirs

        Returns
        -------
        valid_subdirs : list
            Paths to all subdirectories containing valid image/mask pair
        """
        valid_subdirs = []
        if isinstance(self.dirs, list):
            for dir in self.dirs:
                photo = os.path.split(dir)[-1]
                biome_name = os.path.split(os.path.split(dir)[-2])[-1]
                print(f"Parsing Photo: {biome_name}_{photo}")
                for root, dirs, _ in os.walk(dir):
                    valid_subdirs += [os.path.join(root, d) for d in dirs
                                      if os.path.isfile(os.path.join(root, d, 'image.npy'))
                                      and os.path.isfile(os.path.join(root, d, 'mask.npy'))]

        else:
            for root, dirs, _ in os.walk(self.dirs):
                valid_subdirs += [os.path.join(root, d) for d in dirs
                                  if os.path.isfile(os.path.join(root, d, 'image.npy'))
                                  and os.path.isfile(os.path.join(root, d, 'mask.npy'))]
        return valid_subdirs

    def randomly_reduce(self, factor):
        """
        Updates self.paths to random sample of original, with count len(self)*factor
        """
        new_length = int(len(self) * factor)
        self.paths = np.random.choice(self.paths, new_length, replace=False)
        return


class LandsatDataset(Dataset):
    """
    Extension of Dataset class specifically for Landsat data. Includes channel-wise
    averaging and display functions.
    """

    def channel_means(self, samplesize=1.):
        """
        Calculate mean value of each band in dataset's images

        Parameters
        ----------
        samplesize : float, optional
            Factor of sampled pixels, set between 0->1.

        Returns
        -------
        mean : list
            Mean average values in each spectral band
        """
        total_pix = 0
        mean = 0
        frequency = int(1 / samplesize)
        for i in range(0, len(self), frequency):
            im, _ = self[i]
            mean_i = im.mean(axis=(0, 1))
            pix_i = im.shape[0] * im.shape[1]
            mean = np.divide(mean * total_pix + mean_i *
                             pix_i, total_pix + pix_i)
            total_pix += pix_i

        return mean

    def display(self, im, mask, rgb_selection=[3, 2, 1]):
        """
        Create viewable RGB image/mask pair from original

        Parameters
        ----------
        im : array
            Original multispectral image
        mask : array
            Original one-hot encoded mask
        rgb_selection : list, optional
            Specify which bands correspond to Red, Green, Blue  for output


        Returns
        -------
        im : array
            N-by-M-by-3 array with renormalised bands
        mask : array
            N-by-M array with different values for each class
        """
        assert len(rgb_selection) == 3, 'rgb_selection must have length 3'
        im = (im[..., rgb_selection] + 1.2) / 4  # 4 standard deviations from 0->1
        im = np.clip(im, 0, 1)
        mask = np.argmax(mask, axis=-1)
        return im, mask


def train_valid_test(big_dir, train_ratio=0.7, test_ratio=0.1, dataset='Biome', only_test=False, no_test=False):
    bioms_names = ['Barren', 'Forest', 'Grass_Crops', 'Shrubland', 'Snow_Ice', 'Urban', 'Water', 'Wetlands']  # _ can be changed
    train_set = []
    validation_set = []
    test_set = []
    folders = []

    if only_test and no_test:
        raise ValueError('Choose correct parameters for test set: only_test cannot be True if no_test is True.')
    if train_ratio + test_ratio >= 1:
        raise ValueError('Choose correct parameters for test and train ratio.')

    if only_test:
        train_ratio = 0
        test_ratio = 1
    elif no_test:
        test_ratio = 0

    if dataset == 'Biome':
        for dir in os.listdir(big_dir):
            if any(biom in dir for biom in bioms_names):
                path_t0_biom = os.path.join(big_dir, dir)
                folders += [os.path.join(path_t0_biom, f) for f in os.listdir(path_t0_biom) if
                            os.path.isdir(os.path.join(path_t0_biom, f))]

    else:
        folders = [os.path.join(big_dir, folder) for folder in os.listdir(big_dir) if
                   os.path.isdir(os.path.join(big_dir, folder))]

    shuffle(folders)
    total_folders = len(folders)

    if total_folders == 0:
        print("No folders found.")
        return train_set, validation_set, test_set

    if only_test:
        test_set.extend(folders)
        print(f"Testing folders: {len(test_set)}")
        return train_set, validation_set, test_set

    if total_folders == 1:
        train_set.append(folders[0])

    elif total_folders < 4:
        train_set.extend(folders[:-1])
        validation_set.append(folders[-1])

    else:  # total_folders >= 4
        split_train = math.ceil(total_folders * train_ratio)
        split_test = math.ceil(total_folders * test_ratio)

        train_set.extend(folders[:split_train])
        if test_ratio > 0:
            test_set.extend(folders[split_train:split_train + split_test])
        validation_set.extend(folders[split_train + split_test:])

    print(f"Training folders: {len(train_set)}")
    print(f"Testing folders: {len(test_set)}")
    print(f"Validation folders: {len(validation_set)}")

    return train_set, validation_set, test_set


import random


def train_valid_test_sentinel(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, shuffle=True):
    """
    Splits a dataset into training, validation, and test sets based on specified ratios.

    Parameters
    ----------
    dataset : list
        A list of data items (e.g., tuples like (image, mask)).
    train_ratio : float
        The proportion of data for the training set (default is 0.7).
    val_ratio : float
        The proportion of data for the validation set (default is 0.15).
    test_ratio : float
        The proportion of data for the test set (default is 0.15).
    shuffle : bool
        Whether to shuffle the dataset before splitting (default is True).

    Returns
    -------
    train_set : list
        The training dataset.
    val_set : list
        The validation dataset.
    test_set : list
        The test dataset.
    """

    # Validate that the ratios are in the correct range
    if not (0 <= train_ratio <= 1 and 0 <= val_ratio <= 1 and 0 <= test_ratio <= 1):
        raise ValueError("Ratios must be in the range from 0 to 1.")
    if abs(train_ratio + val_ratio + test_ratio - 1) > 1e-6:
        raise ValueError("The sum of the ratios must be equal to 1.")

    # Optionally shuffle the dataset to ensure randomness
    if shuffle:
        random.shuffle(dataset)

    # Calculate the sizes of each subset
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    # Split the dataset into train, validation, and test sets
    train_set = dataset[:train_size]
    val_set = dataset[train_size:train_size + val_size]
    test_set = dataset[train_size + val_size:]

    return train_set, val_set, test_set



def randomly_reduce_list(paths, factor):
    """
    Randomly reduces the list of paths to a sample of the original, with count len(paths) * factor.
    """
    new_length = int(len(paths) * factor)

    if new_length > len(paths):
        new_length = len(paths)
    elif new_length < 1:
        new_length = 1

    sampled_indices = np.random.choice(len(paths), new_length, replace=False)

    sampled_paths = [paths[i] for i in sampled_indices]

    return sampled_paths


if __name__ == '__main__':
    import sys

    # dataset_path = sys.argv[1]
    # dataset = LandsatDataset(dataset_path)
    # dataset.randomly_reduce(0.1)
    # mn = dataset.channel_means()
    # print(mn)
    train_set, validation_set, test_set = train_valid_test("/Volumes/Vault/Splited_data_set_2",
                                                           train_ratio=0.7,
                                                           # test_ratio=0.1,
                                                           dataset='Lan2',
                                                           only_test=True,
                                                           no_test=False)
    # print(test_set)
