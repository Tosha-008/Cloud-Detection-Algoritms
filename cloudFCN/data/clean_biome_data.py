import spectral as spy
import tifffile as tif
import numpy as np
import sys
import os
import ast
from skimage import transform

import Constants


"""
Script for cleaning biome dataset, downloaded from www.usgs.gov

Takes 11-band tiffs plus an extra band for no-data values, and converts into 3D numpy array.
Takes mask envi file and converts to numpy arrays.
Arrays are saved as 'image.npy' and 'mask.npy' in one folder per image
"""

def is_valid_dir(dir):
    """
    Return bool. If dir contains all band files and a .img envi mask file, and no subdirectories
    """
    children = os.listdir(dir)
    if any(os.path.isdir(child) for child in children):
        return False
    for i in range(1, 12):
        suffix = '_B'+str(i)+'.TIF'
        if not any(child.endswith(suffix) for child in children):
            return False
    if not any(child.endswith('fixedmask.img') for child in children):

        return False

    return True


def clean_tile(tile_dir, bands=None, nodata_layer=False, downsample=None):
    # Load constants for band-wise normalisation
    consts = Constants.Landsat_8_constants()
    if bands is None:
        bands = [i for i in range(1, 12)]  # take all 11 bands

    img_arr = None
    for i, band in enumerate(bands):
        print(f'Loading band {band}')
        suffix = '_B' + str(band) + '.TIF'
        path = [os.path.join(tile_dir, f)
                for f in os.listdir(tile_dir) if f.endswith(suffix)]
        if not path:
            print(f'File for band {band} not found.')
            continue
        print(f'File path: {path}')
        band_im = tif.imread(path).astype(np.float32)
        print(f'Image size for band {band}: {band_im.shape}')
        if band == 8 and img_arr is not None:
            band_im = transform.resize(
                band_im, img_arr.shape[0:2], anti_aliasing=True, mode='constant')
        if img_arr is None:  # initialise with shape
            img_arr = np.empty((band_im.shape[0], band_im.shape[1], len(bands)))
            print(f'Initialized img_arr with size {img_arr.shape}')
        img_arr[..., i] = np.squeeze(band_im)

    print(f"All bands loaded - {img_arr.shape} , starting mask processing...")
    print(len(bands))

    # Load mask
    mask_path = [os.path.join(tile_dir, f) for f in os.listdir(tile_dir) if f.endswith('fixedmask.hdr')][0]
    print(f'Mask path: {mask_path}')

    try:
        mask_arr = np.squeeze(spy.open_image(mask_path).load())
        print(f'Mask size: {mask_arr.shape}')
    except Exception as e:
        print(f'Error loading mask: {e}')
        sys.exit(1)

    # Normalize mask
    print('Normalizing mask...')
    mask_arr = mask_arr / 255
    mask_arr = transform.resize(mask_arr, img_arr.shape[0:2], order=0) * 255
    mask_arr = mask_arr.astype('int')

    # Normalize image
    print('Starting image normalization...')
    img_arr = consts.normalise(img_arr, bands)
    print('Image normalization complete')

    if nodata_layer:
        print('Adding NoData layer...')
        img_arr[mask_arr == 0] = 0
        img_arr[..., -1] = mask_arr == 0

    print('Returning processed data...')
    print(f'Image size after normalization: {img_arr.shape}')
    return img_arr, mask_arr



def split_and_save(im, mask, dir, splitsize=398, nodata_amount=0.0):

    n_x = im.shape[0] // splitsize
    n_y = im.shape[1] // splitsize

    for i in range(n_x):
        for j in range(n_y):
            # Split the image and mask into patches
            splitim = im[i * splitsize:(i + 1) * splitsize, j * splitsize:(j + 1) * splitsize, ...]
            splitmaskflat = mask[i * splitsize:(i + 1) * splitsize, j * splitsize:(j + 1) * splitsize, ...]

            print(f"Splitting image {i}, {j}...")

            # Check the size of the split images and masks
            if splitim.shape != (splitsize, splitsize, im.shape[2]):
                raise ValueError(f"Image size error: expected ({splitsize}, {splitsize}, {im.shape[2]}), "
                                 f"but got {splitim.shape}")
            if splitmaskflat.shape != (splitsize, splitsize):
                raise ValueError(f"Mask size error: expected ({splitsize}, {splitsize}), "
                                 f"but got {splitmaskflat.shape}")

            # Convert the mask to one-hot format
            splitmask = np.empty((splitsize, splitsize, 5), dtype=bool)
            splitmask[..., 0] = splitmaskflat == 0  # FILL
            splitmask[..., 1] = splitmaskflat == 64  # SHADOW
            splitmask[..., 2] = splitmaskflat == 128  # CLEAR
            splitmask[..., 3] = splitmaskflat == 192  # THIN
            splitmask[..., 4] = splitmaskflat == 255  # THICK

            # Save only those images that meet the nodata_amount condition
            if np.mean(splitim[..., -1]) <= nodata_amount:
                # Create a directory for saving
                os.makedirs(os.path.join(dir, str(i).zfill(3) + str(j).zfill(3)), exist_ok=True)
                # Save the image and mask in numpy format
                np.save(os.path.join(dir, str(i).zfill(3) + str(j).zfill(3), 'image.npy'), splitim.astype(np.float32))
                np.save(os.path.join(dir, str(i).zfill(3) + str(j).zfill(3), 'mask.npy'), splitmask)

                print(f"Image {i}, {j} saved.")

            print('ROW:', str(i).zfill(2), '  COL:', str(j).zfill(2), end='\r')



if __name__ == '__main__':
    import argparse
    import shutil

    arg_parser = argparse.ArgumentParser(
        description='Clean Biome data for use as input into Fully Convolutional Networks')
    arg_parser.add_argument('biome_data_path',
                            help='path to original biome data (downloadable from https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data)')
    arg_parser.add_argument('out_path', help='path to output directory')
    arg_parser.add_argument('-d', '--downsample', type=float,
                            help='Downsampling factor', default=None)
    arg_parser.add_argument('-s', '--splitsize', type=int,
                            help='Size of each outputted patch', default=250)
    arg_parser.add_argument('-n', '--nodata', type=bool,
                            help='Whether to include a no-data layer in output', default=False)
    arg_parser.add_argument('-t', '--nodata_threshold', type=float,
                            help='Fraction of patch that can have no-data values and still be used',
                            default=1.)
    arg_parser.add_argument('-b', '--bands', type=float,
                            help='List of bands to be used', default=None)
    args = vars(arg_parser.parse_args())

    biome_data_path = args['biome_data_path']
    out_path = args['out_path']
    downsample = args['downsample']
    splitsize = args['splitsize']
    nodata = args['nodata']
    nodata_threshold = args['nodata_threshold']
    bands = args['bands']

    while os.path.isdir(out_path):
        inp = input(
            'Output directory already exists. Continue and overwrite? (Y/N)')
        if inp == 'Y' or inp == 'y':
            shutil.rmtree(out_path)
        elif inp == 'N' or inp == 'n':
            sys.exit()
    os.makedirs(out_path)

    tile_dirs = [os.path.join(biome_data_path, f) for f in os.listdir(
        biome_data_path) if os.path.isdir(os.path.join(biome_data_path, f))]
    for tile_dir in tile_dirs:
        if is_valid_dir(tile_dir):
            print('Loading', os.path.split(tile_dir)[-1], '...')
            img_arr, mask_arr = clean_tile(
                tile_dir, bands=bands, nodata_layer=nodata, downsample=downsample)
            print('\rLoaded', os.path.split(tile_dir)[-1])

            tile_out_path = os.path.join(out_path, os.path.split(tile_dir)[-1])
            os.makedirs(tile_out_path)

            print('Splitting and saving', os.path.split(tile_dir)[-1], '...')
            split_and_save(img_arr, mask_arr, tile_out_path,
                           splitsize, nodata_amount=nodata_threshold)
            print(f"Splitting and saving for {os.path.split(tile_dir)[-1]} completed.")

        else:
            print(tile_dir, ' is not a valid biome dataset directory, skipping...')
