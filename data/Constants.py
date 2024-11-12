import numpy as np
import collections

class Constants():
    """
    A class used to hold constants about bands of a satellite image.

    Attributes
    ----------
    bands_mean : OrderedDict
        Mean pixel values from Landsat 8 for each band number.
    bands_standdev : OrderedDict
        Standard deviation of pixel values from Landsat 8 for each band number.
    """

    def __init__(self):
        self.bands_mean = collections.OrderedDict([
            (1, 0),
            (2, 0),
            (3, 0)
        ])
        self.bands_standdev = collections.OrderedDict([
            (1, 1),
            (2, 1),
            (3, 1)
        ])
        self.band_names = list(self.bands_mean.keys())  # Ensure band_names is defined

    def normalise(self, im, bands=None):
        """
        Channel-wise normalisation of image, subtracting mean of dataset
        and then dividing by its standard deviation

        Parameters
        ----------
        im : array
            3-dimensional array to be normalised.
        bands : list, optional
            Corresponding band designations of image

        Returns
        -------
        im : array
            Normalised image
        """
        if bands is None:
            bands = self.band_names
        elif isinstance(bands, int):
            bands = [bands]

        # Ensure bands are in the correct range
        for band in bands:
            if band not in self.bands_mean:
                raise ValueError(f"Band {band} is not defined in the constants.")

        # Check if the image dimensions match the number of bands
        if im.shape[-1] != len(bands):
            raise ValueError(
                f"Number of bands in image ({im.shape[-1]}) does not match the provided bands list ({len(bands)}).")

        # Check for NaN values in the image and handle them
        if np.isnan(im).any():
            print("Warning: Image contains NaN values. Replacing NaNs with zeros.")
            im = np.nan_to_num(im)

        print(f"Normalizing {len(bands)} bands...")
        for i, band in enumerate(bands):
            if band in self.bands_mean:
                mean = self.bands_mean[band]
                stddev = self.bands_standdev[band]
                if stddev == 0:
                    raise ValueError(f"Standard deviation for band {band} is zero, cannot normalize.")
                print(f"Band {band}: mean={mean}, stddev={stddev}")
                im[..., i] = (im[..., i] - mean) / stddev
            else:
                raise ValueError(f"Band {band} not found in constants.")

        print("Normalization complete.")
        return im


class Landsat_8_constants(Constants):
    """
    A class used to hold constants about Landsat 8 data.

    Attributes
    ----------
    bands_mean : OrderedDict
        Mean pixel values from Landsat 8 for each band number.
    bands_standdev : OrderedDict
        Standard deviation of pixel values from Landsat 8 for each band number.
    """

    def __init__(self):
        super().__init__()

        self.bands_mean = collections.OrderedDict([
            (1, 19940),
            (2, 19620),
            (3, 18720),
            (4, 19190),
            (5, 21880),
            (6, 14030),
            (7, 12170),
            (8, 18860),
            (9, 6430),
            (10, 18520),
            (11, 17110)
        ])

        self.bands_standdev = collections.OrderedDict([
            (1, 11370),
            (2, 11640),
            (3, 11200),
            (4, 11720),
            (5, 11310),
            (6, 6960),
            (7, 5570),
            (8, 11400),
            (9, 3560),
            (10, 8430),
            (11, 7440)
        ])

    def __str__(self):
        return ("Object containing spectral band statistics "
                "(mean and standard deviation) for Landsat 8. "
                "These values were derived from the Biome "
                "dataset, available from the USGS.")
