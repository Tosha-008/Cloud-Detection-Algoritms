import numpy as np
import os
from scipy import misc
import time
import random

def train_base(patch_size, fixed=False):
    """
    Makes a transformation for image/mask/descriptor triplet that is a randomly cropped,
    rotated, and flipped portion of the original.

    Parameters
    ----------
    patch_size : int
        Spatial dimension of the output image/mask pair (assumes Width == Height).
    fixed : bool, optional
        If True, always take the patch from the top-left of the scene, with no rotation
        or flipping. This is useful for validation and reproducibility.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask/descriptor triplets.
    """

    def apply_transform(img, mask, descriptors=None, band_policy=None):
        crop_size = patch_size

        if img.shape[0] < crop_size or img.shape[1] < crop_size:
            return img, mask, descriptors if descriptors is not None else img, mask

        if fixed:
            left = 0
            top = 0
        else:
            left = random.randint(0, img.shape[1] - crop_size)
            top = random.randint(0, img.shape[0] - crop_size)

        img = img[top:min(top + crop_size, img.shape[0]), left:min(left + crop_size, img.shape[1]), ...]
        mask = mask[top:min(top + crop_size, img.shape[0]), left:min(left + crop_size, img.shape[1]), ...]

        rota = random.choice([0, 1, 2, 3])
        flip = random.choice([True, False])
        if rota and not fixed:
            img = np.rot90(img, k=rota)
            mask = np.rot90(mask, k=rota)
        if flip and not fixed:
            img = np.fliplr(img)
            mask = np.fliplr(mask)

        if img.shape[0] != crop_size or img.shape[1] != crop_size:
            print(f"Warning: the final image size is {img.shape}, but ({crop_size}, {crop_size}) was expected")

        return (img, mask, descriptors) if descriptors is not None else (img, mask)


    return apply_transform


# def band_select(bands):
#     """
#     Return image/mask pair where spectral bands in image have been selected as a list.
#
#     Parameters
#     ----------
#     bands : list
#         Spectral bands, defined with respect to input's final dimension.
#
#     Returns
#     -------
#     apply_transform : func
#         Transformation for image/mask pairs.
#     """
#     def apply_transform(img, mask):
#         if bands is not None:
#             img = img[..., bands]
#         return img, mask
#     return apply_transform

def band_select():
    """
    Return a transformation function for selecting spectral bands in an image
    and extracting descriptors, with optional padding.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs and optionally descriptors.
    """
    def apply_transform(img, mask, descriptors=None, band_policy=None):
        """
        Apply the band selection transformation to an image and mask pair.

        Parameters
        ----------
        ...
        band_policy : str / list / int / tuple
            Specifies how bands should be selected.

        Returns
        -------
        ...
        """
        if band_policy is None:
            raise ValueError("band_policy must be specified.")

        padding = 0  # Default no padding
        if band_policy == "all":
            selected_indices = list(range(img.shape[-1]))
        elif band_policy == "RGB":
            selected_indices = [3, 2, 1]  # Assuming RGB corresponds to the first three bands
        elif isinstance(band_policy, list):
            selected_indices = band_policy
        elif isinstance(band_policy, int):
            available_indices = list(range(img.shape[-1]))
            if band_policy <= len(available_indices):
                selected_indices = np.random.choice(available_indices, band_policy, replace=False)
            else:
                selected_indices = available_indices
                padding = band_policy - len(available_indices)
        else:
            raise ValueError(f"Unrecognized band policy: {band_policy}")

        # Select bands from the image
        img = img[..., selected_indices]

        # Optionally, extract descriptors
        selected_descriptors = None
        if descriptors is not None:
            selected_descriptors = [descriptors[i] for i in selected_indices]

        # Apply padding if needed
        if padding > 0:
            img = np.concatenate([img, -0.5 * np.ones((*img.shape[:-1], padding))], axis=-1)
            if selected_descriptors is not None:
                selected_descriptors = np.concatenate([selected_descriptors, -np.ones((padding, *descriptors.shape[1:]))], axis=0)

        return (img, mask, selected_descriptors) if selected_descriptors is not None else (img, mask)

    return apply_transform



def class_merge(class1, class2):
    """
    Create image/mask pairs where classes in mask have been merged (reduces final mask
    dimension by 1).

    Parameters
    ----------
    class1 : int
        Index of class to be fused.
    class2 : int
        Index of class to be fused.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """

    def apply_transform(img, mask, descriptors=None, band_policy=None):
        mask[..., class1] += mask[..., class2]
        mask = mask[..., np.arange(mask.shape[-1]) != class2]
        return (img, mask, descriptors) if descriptors is not None else (img, mask)
    return apply_transform

def sometimes(p, transform):
    """
    Wrapper function which randomly applies the transform with probability p.

    Parameters
    ----------
    p : float
        Probability of transform being applied
    transform : func
        Function which transforms image/mask pairs.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """

    def apply_transform(img, mask, descriptors=None, band_policy=None):
        random_apply = random.random() < p
        if random_apply:
            return transform(img, mask, descriptors) if descriptors is not None else transform(img, mask)
        else:
            return (img, mask, descriptors) if descriptors is not None else (img, mask)
    return apply_transform


def chromatic_shift(shift_min=-0.10, shift_max=0.10):
    """
    Adds a different random amount to each spectral band in image.

    Parameters
    ----------
    shift_min : float, optional
        Lower bound for random shift.
    shift_max : float, optional
        Upper bound for random shift.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """

    def apply_transform(img, mask, descriptors=None, band_policy=None):
        img = img + np.random.uniform(low=shift_min,
                                      high=shift_max, size=[1, 1, img.shape[-1]]).astype(np.float32)
        return (img, mask, descriptors) if descriptors is not None else (img, mask)
    return apply_transform


def chromatic_scale(factor_min=0.90, factor_max=1.10):
    """
    Multiplies each spectral band in batch by a different random factor.

    Parameters
    ----------
    factor_min : float, optional
        Lower bound for random factor.
    factor_max : float, optional
        Upper bound for random factor.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """
    def apply_transform(img, mask, descriptors=None, band_policy=None):
        img = img * np.random.uniform(low=factor_min,
                                      high=factor_max, size=[1, 1, img.shape[-1]]).astype(np.float32)
        return (img, mask, descriptors) if descriptors is not None else (img, mask)
    return apply_transform


def intensity_shift(shift_min=-0.10, shift_max=0.10):
    """
    Adds single random amount to all spectral bands.

    Parameters
    ----------
    shift_min : float, optional
        Lower bound for random shift.
    shift_max : float, optional
        Upper bound for random shift.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """
    def apply_transform(img, mask, descriptors=None, band_policy=None):
        img = img + (shift_max-shift_min)*random.random()+shift_min
        return (img, mask, descriptors) if descriptors is not None else (img, mask)
    return apply_transform


def intensity_scale(factor_min=0.95, factor_max=1.05):
    """
    Multiplies all spectral bands by a single random factor.

    Parameters
    ----------
    factor_min : float, optional
        Lower bound for random factor.
    factor_max : float, optional
        Upper bound for random factor.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """
    def apply_transform(img, mask, descriptors=None, band_policy=None):
        img = img * random.uniform(factor_min, factor_max)
        return (img, mask, descriptors) if descriptors is not None else (img, mask)
    return apply_transform


def white_noise(sigma=0.1):
    """
    Adds white noise to image.

    Parameters
    ----------
    sigma : float, optional
        Standard deviation of white noise

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """

    def apply_transform(img, mask, descriptors=None, band_policy=None):
        noise = (np.random.randn(*img.shape) * sigma).astype(np.float32)
        return (img + noise, mask, descriptors) if descriptors is not None else (img + noise, mask)
    return apply_transform


def bandwise_salt_and_pepper(salt_rate, pepp_rate, pepp_value=0, salt_value=255):
    """
    Adds salt and pepper (light and dark) noise to image,  treating each band independently.

    Parameters
    ----------
    salt_rate : float
        Percentage of pixels that are set to salt_value.
    pepp_rate : float
        Percentage of pixels that are set to pepp_value.
    pepp_value : float, optional
        Value that pepper pixels are set to.
    salt_value : float, optional
        Value that salt pixels are set to.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """
    def apply_transform(img, mask, descriptors=None, band_policy=None):
        salt_mask = np.random.choice([False, True], size=img.shape, p=[
                                     1 - salt_rate, salt_rate])
        pepp_mask = np.random.choice([False, True], size=img.shape, p=[
                                     1 - pepp_rate, pepp_rate])

        img[salt_mask] = salt_value
        img[pepp_mask] = pepp_value

        return (img, mask, descriptors) if descriptors is not None else (img, mask)
    return apply_transform


def salt_and_pepper(salt_rate, pepp_rate, pepp_value=0, salt_value=255):
    """
    Adds salt and pepper (light and dark) noise to image, to all bands in a pixel.

    Parameters
    ----------
    salt_rate : float
        Percentage of pixels that are set to salt_value.
    pepp_rate : float
        Percentage of pixels that are set to pepp_value.
    pepp_value : float, optional
        Value that pepper pixels are set to.
    salt_value : float, optional
        Value that salt pixels are set to.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """
    def apply_transform(img, mask, descriptors=None, band_policy=None):
        salt_mask = np.random.choice(
            [False, True], size=img.shape[:-1], p=[1 - salt_rate, salt_rate])
        pepp_mask = np.random.choice(
            [False, True], size=img.shape[:-1], p=[1 - pepp_rate, pepp_rate])

        img[salt_mask] = [salt_value for i in range(img.shape[-1])]
        img[pepp_mask] = [pepp_value for i in range(img.shape[-1])]

        return (img, mask, descriptors) if descriptors is not None else (img, mask)
    return apply_transform


def quantize(number_steps, min_value=0, max_value=255, clip=False):
    """
    Quantizes an image based on a given number of steps by rounding values to closest
    value.

    Parameters
    ----------
    number_steps : int
        Number of values to round to
    min_value : float
        Lower bound of quantization
    max_value : float
        Upper bound of quantization
    clip : bool
        True if values outside of [min_value:max_value] are clipped. False otherwise.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """
    stepsize = (max_value-min_value)/number_steps

    def apply_transform(img, mask, descriptors=None, band_policy=None):
        img = (img//stepsize)*stepsize
        if clip:
            img = np.clip(img, min_value, max_value)
        return (img, mask, descriptors) if descriptors is not None else (img, mask)
    return apply_transform


def normalize_to_range(min_value=0.0, max_value=1.0):
    """
    Normalizes the image to the specified range for each channel independently.

    Parameters
    ----------
    min_value : float, optional
        Minimum value of the normalized output.
    max_value : float, optional
        Maximum value of the normalized output.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """

    def apply_transform(img, mask, descriptors=None, band_policy=None):
        """
        Applies normalization to the image.

        Parameters
        ----------
        img : numpy.ndarray
            Input image array with shape (H, W, C), where C is the number of channels.
        mask : numpy.ndarray
            Corresponding mask array, which remains unchanged.

        Returns
        -------
        normalized_image : numpy.ndarray
            Image normalized to the specified range.
        mask : numpy.ndarray
            Unchanged mask.
        """
        img = img.astype(np.float32)  # Ensure compatibility with floating-point operations
        channel_min = np.min(img, axis=(0, 1), keepdims=True)  # Per-channel minimum
        channel_max = np.max(img, axis=(0, 1), keepdims=True)  # Per-channel maximum

        # Prevent division by zero
        denominator = np.maximum(channel_max - channel_min, 1e-6)
        normalized_image = (img - channel_min) / denominator

        # Scale to [min_value, max_value]
        normalized_image = normalized_image * (max_value - min_value) + min_value

        return (normalized_image, mask, descriptors) if descriptors is not None else (normalized_image, mask)

    return apply_transform


def sentinel_13_to_11():
    def apply_transform(img, mask, descriptors=None, band_policy=None):

        if img.shape[-1] != 13:
            print(f"Skipping: expected 13 channels, found {img.shape[-1]}")
            pass
        height, width = img.shape[:2]
        black_layer = np.zeros((height, width, 1), dtype=img.dtype)
        avg_B5_B6_B7 = (img[:, :, 4] + img[:, :, 5] + img[:, :, 6]) / 3

        selected_channels = [
            img[:, :, 3],  # Band 4 (Red)
            img[:, :, 2],  # Band 3 (Green)
            img[:, :, 1],  # Band 2 (Blue)
            img[:, :, 0],  # Band 1 (Coastal Aerosol)
            img[:, :, 8],  # Band 8A (Vegetation Red Edge)
            img[:, :, 11],  # Band 11 (SWIR 1)
            img[:, :, 12],  # Band 12 (SWIR 2)
            avg_B5_B6_B7,  # B5, B6, B7
            img[:, :, 10],  # Band 10 (Cirrus)
            img[:, :, 7],  # Band 8 (NIR)
            img[:, :, 9]  # Band 9 (Water Vapor)
        ]

        final_image = np.stack(selected_channels, axis=-1)
        img = np.concatenate((final_image, black_layer), axis=-1)  # Nodata layer
        return (img, mask, descriptors) if descriptors is not None else (img, mask)
    return apply_transform


def landsat_12_to_13():
    def apply_transform(img, mask, descriptors=None, band_policy=None):
        if img.shape[-1] != 12:
            print(f"Skipping: expected 11 channels, found {img.shape[-1]}")
            return None

        red_edge_sim = (img[:, :, 3] + img[:, :, 4]) / 2  # Аппроксимация Red Edge (B5, B6, B7)
        vapor_sim = (img[:, :, 4] - img[:, :, 6]) / (img[:, :, 4] + img[:, :, 6] + 1e-6)
        vapor_sim_norm = (vapor_sim - np.min(vapor_sim)) / (np.max(vapor_sim) - np.min(vapor_sim) + 1e-6)

        cirrus_sim = img[:, :, 8]  # Используем Thermal Band (B9) как Cirrus
        additional_red_edge = red_edge_sim  # Ещё один Red Edge для B8A

        selected_channels_1 = [
            img[:, :, 3],  # Band 4 (Red) -> Sentinel B4
            img[:, :, 2],  # Band 3 (Green) -> Sentinel B3
            img[:, :, 1],  # Band 2 (Blue) -> Sentinel B2
            img[:, :, 0],  # Band 1 (Coastal Aerosol) -> Sentinel B1
            red_edge_sim,  # Simulated Red Edge -> Sentinel B5
            red_edge_sim,  # Simulated Red Edge -> Sentinel B6
            red_edge_sim,  # Simulated Red Edge -> Sentinel B7
            img[:, :, 4],  # Band 5 (NIR) -> Sentinel B8
            additional_red_edge,  # Additional Red Edge -> Sentinel B8A
            vapor_sim_norm,  # Simulated Water Vapor -> Sentinel B9
            cirrus_sim,  # Simulated Cirrus -> Sentinel B10
            img[:, :, 7],  # Band 6 (SWIR 1) -> Sentinel B11
            img[:, :, 8],  # Band 7 (SWIR 2) -> Sentinel B12
        ]

        final_image = np.stack(selected_channels_1, axis=-1)
        # for i in range(final_image.shape[-1]):
        #     print(f"Channel {i}: {final_image[:, :, i].min()} - {final_image[:, :, i].max()}")

        return (final_image, mask, descriptors) if descriptors is not None else (final_image, mask)
    return apply_transform

def change_mask_channels_2_3():
    """Combines mask channels based on the dataset type."""
    def apply_transform(img, mask, descriptors=None, band_policy=None):
        combined = np.zeros((mask.shape[0], mask.shape[1], 3))
        combined[:, :, 0] = mask[:, :, 0]
        combined[:, :, 1] = mask[:, :, 2]  # Swap channels 2 and 3
        combined[:, :, 2] = mask[:, :, 1]
        return (img, combined, descriptors) if descriptors is not None else (img, combined)
    return apply_transform


def encode_descriptors(descriptor_style):
    """
    Returns a transformation function for encoding descriptors based on a given style.

    Parameters
    ----------
    descriptor_style : str
        Encoding style for the descriptors. Options:
        - 'bandpass': Applies bandpass encoding.
        - 'log': Applies logarithmic encoding.

    Returns
    -------
    apply_transform : func
        A transformation function that encodes descriptors based on the specified style.
    """
    def apply_transform(img, mask, descriptors, band_policy=None):
        """
        Encodes the descriptors based on the specified style.

        Parameters
        ----------
        img : np.ndarray
            Image data (unchanged in this transformation).
        mask : np.ndarray
            Mask data (unchanged in this transformation).
        descriptors : np.ndarray
            Descriptors to be encoded.

        Returns
        -------
        img : np.ndarray
            Unchanged image data.
        mask : np.ndarray
            Unchanged mask data.
        descriptors : np.ndarray
            Encoded descriptors.
        """
        descriptors = np.array(descriptors, dtype=np.float32)
        if descriptor_style == 'bandpass':
            centres = descriptors[:, 1]
            widths = 1.5 * np.log((descriptors[:, 2] - descriptors[:, 0]) / centres) / np.log(15) + 1.5

            RGB_min, RGB_max = 400, 680
            RGB_centre = (RGB_max + RGB_min) / 2
            NIR_min, NIR_max = 680, 1100
            NIR_centre = (NIR_max + NIR_min) / 2
            cirr_min, cirr_max = 1100, 1600
            cirr_centre = (cirr_max + cirr_min) / 2
            SWIR_min, SWIR_max = 1100, 3000
            SWIR_centre = (SWIR_max + SWIR_min) / 2
            TIR_min, TIR_max = 9000, 13000
            TIR_centre = (TIR_max + TIR_min) / 2

            RGB_pass = 0.5 + 0.5 * np.tanh(1 * (centres - RGB_centre) / (RGB_max - RGB_centre))
            NIR_pass = 0.5 + 0.5 * np.tanh(1.5 * (centres - NIR_centre) / (NIR_max - NIR_centre))
            cirr_pass = 0.5 + 0.5 * np.tanh(2 * (centres - cirr_centre) / (cirr_max - cirr_centre))
            SWIR_pass = 0.5 + 0.5 * np.tanh(2 * (centres - SWIR_centre) / (SWIR_max - SWIR_centre))
            TIR_pass = 0.5 + 0.5 * np.tanh(2 * (centres - TIR_centre) / (TIR_max - TIR_centre))

            descriptors = np.stack([RGB_pass, NIR_pass, cirr_pass, SWIR_pass, TIR_pass, widths], axis=1)

        elif descriptor_style == 'log':
            # Apply the transformation
            descriptors = np.log10(descriptors - 300) - 2

        else:
            raise ValueError(f"Unsupported descriptor style: {descriptor_style}")

        return img, mask, descriptors

    return apply_transform
