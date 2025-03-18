import os
import numpy as np
from osgeo import gdal


def split_and_save_for_Fmask(input_dir, output_dir, splitsize=398):
    """
    Splits large raster images into smaller tiles for Fmask processing and saves them as .npy files.

    Parameters:
        input_dir (str): Directory containing the input raster images (.img format).
        output_dir (str): Directory where the split tiles will be saved.
        splitsize (int, optional): Size of the square tiles (default is 398x398 pixels).

    Returns:
        None
    """

    # Iterate through all files in the input directory
    for file in os.listdir(input_dir):
        if not file.endswith('.img'):
            continue

        file_path = os.path.join(input_dir, file)
        fmask_dataset = gdal.Open(file_path)

        if not fmask_dataset:
            raise ValueError(f"Failed to open file {file_path}")

        im = fmask_dataset.GetRasterBand(1).ReadAsArray()

        n_x = im.shape[1] // splitsize
        n_y = im.shape[0] // splitsize

        file_name = os.path.basename(file)
        base_name = file_name.split('_')[0]

        base_dir = os.path.join(output_dir, base_name)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        for i in range(n_x):
            for j in range(n_y):
                splitim = im[i * splitsize:(i + 1) * splitsize, j * splitsize:(j + 1) * splitsize]

                split_dir = os.path.join(base_dir, f"{str(i).zfill(3)}_{str(j).zfill(3)}")
                os.makedirs(split_dir, exist_ok=True)

                np.save(os.path.join(split_dir, 'image.npy'), splitim.astype(np.float32))
                print(f'ROW: {str(i).zfill(2)}  COL: {str(j).zfill(2)}', end='\r')

    print("\nProcessing complete.")

if __name__ == '__main__':
    split_and_save_for_Fmask(input_dir = "/Users/tosha_008/Downloads/Fmask_masks/Unsplited",
                             output_dir = "/Users/tosha_008/Downloads/Fmask_masks/Splited",
                             splitsize=384)
