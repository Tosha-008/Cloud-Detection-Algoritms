import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from MFCNN.model_mfcnn_def import *
import pickle
import json
import csv
# import cv2
from MFCNN.model_mfcnn_def import MCDropoutModel


custom_objects = {
    'MultiscaleLayer': MultiscaleLayer,
    'Up': Up,
    'FMM': FMM,
    'PadByUp': PadByUp,
    'DoubleConv': DoubleConv,
    'ScaleBlock': ScaleBlock,
    'PaddingLayer': PaddingLayer,
    'OutConv': OutConv
}

def calculate_ndvi(sentinel_image):
    """
    Calculates NDVI as a proxy for surface temperature using Sentinel-2 data.

    Parameters:
        sentinel_image (np.array): Sentinel-2 image with 13 channels.

    Returns:
        np.array: NDVI index.
    """
    nir = sentinel_image[:, :, 7]  # B8
    red = sentinel_image[:, :, 3]  # B4
    ndvi = (nir - red) / (nir + red)
    return ndvi


def thermal_proxy(sentinel_image):
    """
    Creates a proxy for thermal data using SWIR and NDVI.

    Parameters:
        sentinel_image (np.array): Sentinel-2 image with 13 channels.

    Returns:
        np.array: Thermal proxy.
    """
    swir1 = sentinel_image[:, :, 11]  # B11
    swir2 = sentinel_image[:, :, 12]  # B12
    ndvi = calculate_ndvi(sentinel_image)
    thermal_proxy = (swir1 + swir2) / 2 * (1 - ndvi)
    return thermal_proxy


from scipy.ndimage import zoom


def resample_to_landsat_resolution(channel, sentinel_res=60, landsat_res=30):
    """
    Resamples a Sentinel-2 channel to Landsat 8 resolution.

    Parameters:
        channel (np.array): Single Sentinel-2 channel (60 m resolution).
        sentinel_res (int): Resolution of Sentinel-2 channel in meters.
        landsat_res (int): Desired resolution (Landsat 8 resolution) in meters.

    Returns:
        np.array: Resampled channel.
    """
    scale_factor = sentinel_res / landsat_res
    return zoom(channel, scale_factor, order=1)  # Bilinear interpolation


def landsat_12_to_13(img, variant=1):
    if img.shape[-1] != 12:
        print(f"Skipping: expected 11 channels, found {img.shape[-1]}")
        return None

    red_edge_sim = (img[:, :, 3] + img[:, :, 4]) / 2  # Аппроксимация Red Edge (B5, B6, B7)
    vapor_sim = (img[:, :, 4] - img[:, :, 6]) / (img[:, :, 4] + img[:, :, 6] + 1e-6)
    vapor_sim_norm = (vapor_sim - np.min(vapor_sim)) / (np.max(vapor_sim) - np.min(vapor_sim) + 1e-6)
    cirrus_sim = img[:, :, 8]  # Используем Thermal Band (B9) как Cirrus
    thermal_sim = (img[:, :, 9] + img[:, :, 10]) / 2  # Среднее Thermal для B10
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

    selected_channels_2 = [
        img[:, :, 3],  # Band 2 (Blue)
        img[:, :, 2],  # Band 3 (Green)
        img[:, :, 1],  # Band 4 (Red)
        img[:, :, 0],  # Band 1 (Coastal Aerosol)
        img[:, :, 4],  # Band 5 (NIR)
        img[:, :, 6],  # Band 6 (SWIR 1)
        img[:, :, 7],  # Band 7 (SWIR 2)
        red_edge_sim,  # Simulierter Red Edge
        cirrus_sim,    # Simulierter Cirrus
        vapor_sim,     # Approximierter Wasserdampf
        thermal_sim,   # Simulierter Thermal
        vapor_sim,     # Zusätzlicher simulierter Wasserdampf
        thermal_sim    # Zusätzlicher simulierter Thermal
    ]
    if variant == 1:
        final_image = np.stack(selected_channels_1, axis=-1)
    elif variant == 2:
        final_image = np.stack(selected_channels_2, axis=-1)
    else:
        print(f"Invalid variant: {variant}. Returning None.")
        return None

    return final_image




def sentinel_13_to_11(img, variant=1):
    if img.shape[-1] != 13:
        print(f"Skipping: expected 13 channels, found {img.shape[-1]}")
        pass
    height, width = img.shape[:2]
    black_layer = np.zeros((height, width, 1), dtype=img.dtype)

    avg_B5_B6_B7 = (img[:, :, 4] + img[:, :, 5] + img[:, :, 6]) / 3

    thermal_ = thermal_proxy(img)

    pancrom = (img[:, :, 1] + img[:, :, 2] + img[:, :, 3] + img[:, :, 7]) / 4
    panchromatic_approx = (0.3 * img[:, :, 1] + 0.3 * img[:, :, 2] + 0.3 * img[:, :, 3] + 0.1 * img[:, :, 7])

    # sentinel_b1_resampled = resample_to_landsat_resolution(img[:, :, 0])

    # cirrus_resampled = resample_to_landsat_resolution(img[:, :, 10], sentinel_res=60, landsat_res=30)  # Scale from 60m to 30m


    selected_channels_1 = [
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

    selected_channels_2 = [
        img[:, :, 3],  # Band 4 (Red)
        img[:, :, 2],  # Band 3 (Green)
        img[:, :, 1],  # Band 2 (Blue)
        img[:, :, 0],  # Band 1 (Coastal Aerosol)
        img[:, :, 7],  # Band 8 (NIR)
        img[:, :, 11],  # Band 11 (SWIR 1)
        img[:, :, 12],  # Band 12 (SWIR 2)
        pancrom,  # B1, B2, B3, B8
        img[:, :, 10],  # Band 10 (Cirrus)
        thermal_,  # Band 8 (NIR)
        thermal_  # Band 9 (Water Vapor)
    ]

    selected_channels_3 = [
        img[:, :, 3],  # Band 4 (Red)
        img[:, :, 2],  # Band 3 (Green)
        img[:, :, 1],  # Band 2 (Blue)
        img[:, :, 0],  # Band 1 (Coastal Aerosol)
        img[:, :, 7],  # Band 8 (NIR)
        img[:, :, 11],  # Band 11 (SWIR 1)
        img[:, :, 12],  # Band 12 (SWIR 2)
        panchromatic_approx,  # B1, B2, B3, B8
        img[:, :, 10],  # Band 10 (Cirrus)
        thermal_,  # Band 8 (NIR)
        thermal_  # Band 9 (Water Vapor)
    ]

    if variant == 1:
        final_image = np.stack(selected_channels_1, axis=-1)
    elif variant == 2:
        final_image = np.stack(selected_channels_2, axis=-1)
    elif variant == 3:
        final_image = np.stack(selected_channels_3, axis=-1)

    img = np.concatenate((final_image, black_layer), axis=-1)  # Nodata layer
    return img

def load_and_preprocess(image_path, mask_path, fmask_path, dataset_name, model_name):
    """
    Loads and preprocesses image, mask, and Fmask data.

    Parameters:
        image_path (str): Path to the image file.
        mask_path (str): Path to the mask file.
        fmask_path (str or None): Path to the Fmask file.
        dataset_name (str): Dataset type.

    Returns:
        tuple: Preprocessed (image, mask, binary_fmask).
    """
    # Load data
    image = np.load(image_path)
    mask = np.load(mask_path)
    binary_fmask = None

    if fmask_path and os.path.exists(fmask_path):
        fmask = np.load(fmask_path)
        binary_fmask = (fmask == 2).astype(int)

    # Combine and normalize
    # Prepare the image for prediction
    mask = combine_channels(mask, dataset_name)
    image = reorder_channels(image, dataset_name, model_name)
    image = normalize_image(image, mode=1)
    image_reordered_expanded = np.expand_dims(image, axis=0)

    return image_reordered_expanded, mask, binary_fmask

def process_and_evaluate(pred_mask, mask, binary_fmask=None, model_name='mfcnn', dataset_name='Set_2', min_area=5, min_pixels=17, alpha=None):
    """
    Process the predicted mask and calculate metrics.

    Parameters:
        pred_mask (numpy array): Predicted mask from the model.
        mask (numpy array): Ground truth mask.
        binary_fmask (numpy array, optional): Binary Fmask mask. Defaults to None.
        model_name (str): Name of the model ('cxn', 'mfcnn', etc.).
        dataset_name (str): Name of the dataset ('Set_2', 'Biome', etc.).
        alpha (float): Threshold for binary mask creation.
        min_area (int): Minimum contour area to keep during processing.
        min_pixels (int): Minimum number of positive pixels in the mask to retain.

    Returns:
        dict: Calculated metrics for the predicted mask and Fmask (if provided).
    """
    # Apply threshold based on alpha and model_name
    if alpha is None:
        if dataset_name in ['Set_2', 'Biome'] and model_name in ['mfcnn', 'cxn']:
            alpha = 7.743e-12 if model_name == 'mfcnn' else 0.0004394 if model_name == 'cxn' else 0.5
        elif dataset_name == 'Sentinel_2' and model_name in ['mfcnn', 'cxn']:
            alpha = 0.12 if model_name == 'mfcnn' else 0.61 if model_name == 'cxn' else 0.5
        elif dataset_name in ['Set_2', 'Biome'] and model_name in ['mfcnn_sentinel', 'cxn_sentinel']:
            alpha = 0.25 if model_name == 'mfcnn_sentinel' else 0.5
        elif dataset_name == 'Sentinel_2'and model_name in ['mfcnn_sentinel', 'cxn_sentinel']:
            alpha =  0.58 if model_name == 'mfcnn_sentinel' else 0.5
        elif dataset_name in ['Set_2', 'Biome', 'Sentinel_2'] and model_name in ['mfcnn_finetuned', 'mfcnn_finetuned_lowclouds']:
            alpha = 0.27 if model_name == 'mfcnn_finetuned' else 0.16 if model_name == 'mfcnn_finetuned_lowclouds' else 0.5
        elif dataset_name in ['Set_2', 'Biome'] and model_name in ['mfcnn_common']:
            alpha = 0.33 if model_name == 'mfcnn_common' else 0.16
        elif dataset_name == 'Sentinel_2'and model_name in ['mfcnn_common']:
            alpha =  0.52 if model_name == 'mfcnn_common' else 0.5

    if model_name in ['cxn', 'mfcnn']:
        pred_mask_binary = (pred_mask.squeeze()[:, :, -1] > alpha).astype(float)
    elif model_name in ['cxn_sentinel', 'mfcnn_sentinel', 'mfcnn_finetuned', 'mfcnn_finetuned_lowclouds', 'mfcnn_common']:
        pred_mask_binary = (pred_mask.squeeze()[:, :, -2] > alpha).astype(float)
    if model_name == 'cxn':
        # Postprocess predicted mask
        kernel = np.ones((3, 3), np.uint8)
        pred_mask_binary = cv2.morphologyEx(pred_mask_binary, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(pred_mask_binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output_mask = np.zeros_like(pred_mask_binary, dtype=np.uint8)
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(output_mask, [contour], -1, 1, -1)

        kernel = np.ones((4, 4), np.uint8)
        dilated_mask = cv2.dilate(output_mask, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(output_mask)
        cv2.drawContours(final_mask, contours, -1, 1, -1)

        pred_mask_binary = final_mask

    pred_mask_binary = classify_no_clouds(pred_mask.squeeze(), pred_mask_binary, model_name, using=False)

    # Flatten masks
    mask_flat = mask.squeeze()[:, :, -1].flatten()
    pred_mask_binary_flat = pred_mask_binary.flatten()

    # Handle binary_fmask if provided
    if binary_fmask is not None:
        binary_fmask_flat = binary_fmask.flatten()

    # Filter small predictions
    if np.sum(pred_mask_binary_flat) < min_pixels:
        pred_mask_binary_flat[:] = 0

    # Calculate metrics for predicted mask
    metrics_pred = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
    if dataset_name in ['Set_2', 'Biome', 'Sentinel_2']:
        metrics_pred["accuracy"] = accuracy_score(mask_flat, pred_mask_binary_flat)
        metrics_pred["precision"] = precision_score(mask_flat, pred_mask_binary_flat, zero_division=1)
        metrics_pred["recall"] = recall_score(mask_flat, pred_mask_binary_flat, zero_division=1)
        metrics_pred["f1"] = f1_score(mask_flat, pred_mask_binary_flat, zero_division=1)

    # Calculate metrics for Fmask if provided
    metrics_fmask = None
    if binary_fmask is not None:
        metrics_fmask = {
            "accuracy": accuracy_score(mask_flat, binary_fmask_flat),
            "precision": precision_score(mask_flat, binary_fmask_flat, zero_division=1),
            "recall": recall_score(mask_flat, binary_fmask_flat, zero_division=1),
            "f1": f1_score(mask_flat, binary_fmask_flat, zero_division=1),
        }

    return {"predicted": metrics_pred, "fmask": metrics_fmask}, pred_mask_binary


def adjustment_input(pickle_file, group_name, max_objects=5, shuffle=False):
    """
    Adjusts input data from a pickle file based on the specified group name and other parameters.

    Parameters
    ----------
    pickle_file : str
        Path to the pickle file containing cloudiness groups.
    group_name : str
        Name of the group to retrieve or 'no filter' for all groups.
    max_objects : int, optional
        Maximum number of objects to include (default is 5).
    shuffle : bool, optional
        Whether to shuffle the resulting list (default is False).

    Returns
    -------
    list
        Adjusted list of paths.
    """
    if not pickle_file:
        raise ValueError("A pickle file with cloudiness groups must be provided.")

    # Load data from the pickle file
    with open(pickle_file, "rb") as f:
        cloudiness_groups = pickle.load(f)

    # Handle 'no filter' case
    if group_name == 'no filter':
        if isinstance(cloudiness_groups, dict):
            # Determine the minimum list length across all keys
            min_length = min(len(lst) for lst in cloudiness_groups.values())
            # Adjust min_length based on max_objects
            min_length = min(min_length, max_objects // 5)
            # Combine up to min_length items from each list
            group_folders = [item for lst in cloudiness_groups.values() for item in lst[:min_length]]
        elif isinstance(cloudiness_groups, list):
            group_folders = cloudiness_groups
        else:
            raise ValueError("Unsupported format in pickle file for 'no filter'.")
    else:
        # Retrieve the specific group
        if group_name not in cloudiness_groups:
            raise ValueError(f"Group '{group_name}' not found in pickle file.")
        group_folders = cloudiness_groups[group_name]

    # Shuffle if required
    if shuffle:
        random.shuffle(group_folders)

    # Apply max_objects limit
    group_folders = group_folders[:max_objects] if max_objects else group_folders

    # Validate result
    total_paths = len(group_folders)
    if total_paths == 0:
        raise ValueError(f"No paths found for group '{group_name}'.")

    print(total_paths)
    return group_folders



def aggregate_image_metrics(group_folders, dataset_name, model, model_name, display=False, display_chanel=None,
                            norm_folder=None, fmask_folder=None, alpha=None, predict_uncertainty=False, T=15):
    """
    Calculate metrics for a group of images.

    Parameters:
        group_folders (list): List of folders or image-mask pairs.
        dataset_name (str): Dataset type ('Set_2', 'Biome', etc.).
        norm_folder (str): Path to the normalized folder.
        fmask_folder (str): Path to the Fmask folder.
        model (keras.Model): Loaded model for predictions.
        model_name (str): Name of the model for logging purposes.
        load_and_preprocess (function): Function to load and preprocess images and masks.
        process_and_evaluate (function): Function to compute evaluation metrics.

    Returns:
        dict: Aggregated metrics for predictions and Fmask.
    """

    # Initialize metrics accumulators
    all_metrics_pred = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    all_metrics_fmask = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    total_paths = len(group_folders)

    # Loop through the folders in the specified group
    for idx, folder in enumerate(group_folders, start=1):
        if dataset_name in ['Set_2', 'Biome']:
            folder2_path = os.path.join(norm_folder, folder)
            folder1_name = folder[:-3] + "_" + folder[-3:]
            folder1_path = os.path.join(fmask_folder, folder1_name)

            # Paths to the images and masks
            image_path = os.path.join(folder2_path, 'image.npy')
            mask_path = os.path.join(folder2_path, 'mask.npy')
            fmask_path = os.path.join(folder1_path, 'image.npy')

            if not (os.path.exists(image_path) and os.path.exists(mask_path) and os.path.exists(fmask_path)):
                continue
        else:
            image_path, mask_path = folder
            # root = '/home/ladmin/PycharmProjects/cloudFCN-master'# Can be deleted or changed
            # image_clean_path = image_path.lstrip('./')
            # mask_clean_path = mask_path.lstrip('./')
            # image_path = os.path.join(root, image_clean_path)
            # mask_path = os.path.join(root, mask_clean_path)
            fmask_path = None
            if not (os.path.exists(image_path) and os.path.exists(mask_path)):
                continue

        # Load and preprocess data
        image, mask, binary_fmask = load_and_preprocess(image_path, mask_path, fmask_path, dataset_name, model_name)

        # Predict mask using the model
        if predict_uncertainty:
            mean_pred, std_pred = predict_with_uncertainty(model, image, T=T)
            pred_mask = (mean_pred.squeeze()[:, :, -2] > 0.45).astype(float)
            uncertainty_mask = (std_pred.squeeze()[:, :, -2] > 0.1).astype(float)
        else:
            pred_mask = model.predict(image)
            # Process and evaluate predictions
            metrics, pred_mask_binary = process_and_evaluate(pred_mask, mask, binary_fmask, model_name, dataset_name, alpha=alpha)

            # Accumulate predicted metrics
            all_metrics_pred["accuracy"].append(metrics["predicted"]["accuracy"])
            all_metrics_pred["precision"].append(metrics["predicted"]["precision"])
            all_metrics_pred["recall"].append(metrics["predicted"]["recall"])
            all_metrics_pred["f1"].append(metrics["predicted"]["f1"])

            # Accumulate Fmask metrics if available
            if binary_fmask is not None:
                all_metrics_fmask["accuracy"].append(metrics["fmask"]["accuracy"])
                all_metrics_fmask["precision"].append(metrics["fmask"]["precision"])
                all_metrics_fmask["recall"].append(metrics["fmask"]["recall"])
                all_metrics_fmask["f1"].append(metrics["fmask"]["f1"])

        print(f"Processed {idx}/{total_paths} folders ({(idx / total_paths) * 100:.2f}%)")

        if display:
            # Display the images
            fig, axes = plt.subplots(2, 2, figsize=(18, 12))  # Create 2x2 grid

            image_rgb = image.squeeze()[:, :, :3]
            axes[0, 0].imshow(image_rgb)
            axes[0, 0].set_title(f"Image from {os.path.dirname(image_path)}")
            axes[0, 0].axis('off')

            axes[0, 1].imshow(mask[:, :, -1], cmap='gray', vmin=0, vmax=1)
            axes[0, 1].set_title("Mask")
            axes[0, 1].axis('off')

            if predict_uncertainty:
                axes[1, 0].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
                axes[1, 0].set_title(f"Predicted Mask")
                axes[1, 0].axis('off')

                axes[1, 1].imshow(std_pred.squeeze()[:, :, -2] * 10, cmap='gray', vmin=0, vmax=1)
                axes[1, 1].set_title("Uncertainty Mask")
                axes[1, 1].axis('off')

            else:
                if binary_fmask is not None:
                    if display_chanel:
                        axes[1, 0].imshow(pred_mask.squeeze()[:, :, -display_chanel], cmap='gray', vmin=0, vmax=1)
                    else:
                        axes[1, 0].imshow(binary_fmask, cmap='gray', vmin=0, vmax=1)
                    axes[1, 0].set_title("Fmask")
                    axes[1, 0].axis('off')

                    # Add metrics below the corresponding images
                    metrics_text_fmask = (
                        f'Accuracy: {metrics["fmask"]["accuracy"]:.2f}\n'
                        f'Precision: {metrics["fmask"]["precision"]:.2f}\n'
                        f'Recall: {metrics["fmask"]["recall"]:.2f}\n'
                        f'F1 Score: {metrics["fmask"]["f1"]:.2f}'
                    )
                    axes[1, 0].text(0.5, -0.2, metrics_text_fmask, color='black', ha='center', va='top',
                                    transform=axes[1, 0].transAxes, fontsize=10)

                axes[1, 1].imshow(pred_mask_binary, cmap='gray', vmin=0, vmax=1)
                axes[1, 1].set_title(f'Predicted Mask')
                axes[1, 1].axis('off')

                # Add metrics below the corresponding images
                metrics_text_pred = (
                    f'Accuracy: {metrics["predicted"]["accuracy"]:.2f}\n'
                    f'Precision: {metrics["predicted"]["precision"]:.2f}\n'
                    f'Recall: {metrics["predicted"]["recall"]:.2f}\n'
                    f'F1 Score: {metrics["predicted"]["f1"]:.2f}'
                )
                axes[1, 1].text(0.5, -0.2, metrics_text_pred, color='black', ha='center', va='top',
                                transform=axes[1, 1].transAxes, fontsize=10)

            plt.tight_layout()
            plt.show()

    # Return aggregated metrics
    return None if predict_uncertainty else {
        "predicted": {k: sum(v) / len(v) for k, v in all_metrics_pred.items() if v},
        "fmask": {k: sum(v) / len(v) for k, v in all_metrics_fmask.items() if v} if all_metrics_fmask["accuracy"] else None
    }




def split_images_by_cloudiness(folder, output_file, dataset_name='Set_2', mask_storage=None):
    """
    Function to iterate over images in a dataset, calculate their cloudiness levels,
    and split them into separate lists by cloudiness grade.

    Parameters:
        folder (str or list): Path to the folder containing subfolders with images and masks,
                              or a list of tuples (image_path, mask_path) for Sentinel dataset.
        output_file (str): Path to save the resulting cloudiness lists as a pickle file.
        dataset_name (str): Dataset type, e.g., 'Set_2', 'Biome', or 'Sentinel'.

    Returns:
        dict: Dictionary containing lists of image paths categorized by cloudiness grade.
    """
    cloudiness_groups = {
        "low": [],  # 0-30%
        "middle": [],  # 30-70%
        "high": [],  # 70-100%
        "only clouds": [],  # 100%
        "no clouds": []  # 0%
    }

    if isinstance(folder, list):  # Sentinel dataset
        file_pairs = folder
    else:  # Other datasets
        if mask_storage:
            with open(mask_storage, "rb") as f:
                norm_subfolders = pickle.load(f)
        else:
            norm_subfolders = get_subfolders(folder, 'norm_subfolders')

        file_pairs = []
        for path in norm_subfolders:
            subfolder = os.path.join(folder, path)
            image_path = os.path.join(subfolder, 'image.npy')
            mask_path = os.path.join(subfolder, 'mask.npy')
            if os.path.exists(image_path) and os.path.exists(mask_path):
                file_pairs.append((image_path, mask_path))

    total_files = len(file_pairs)

    for idx, (image_path, mask_path) in enumerate(file_pairs):
        # root = '/home/ladmin/PycharmProjects/cloudFCN-master'  #Can be deleted or changed
        # image_clean_path = image_path.lstrip('./')
        # mask_clean_path = mask_path.lstrip('./')
        # image_path = os.path.join(root, image_clean_path)
        # mask_path = os.path.join(root, mask_clean_path)
        if not os.path.exists(mask_path):
            continue

        # Load the mask
        mask = np.load(mask_path)

        # Combine channels based on dataset type
        mask = combine_channels(mask, dataset_name)

        # Calculate the cloudiness level
        cloud_pixels = np.sum(mask[:, :, -1] > 0)  # Assuming cloud pixels are marked with values > 0
        total_pixels = mask[:, :, -1].size
        cloud_coverage = (cloud_pixels / total_pixels) * 100

        if dataset_name in ['Set_2', 'Biome']:
            path_to_add = os.path.dirname(image_path)
        else:
            path_to_add = (image_path, mask_path)

        # Categorize the image by cloudiness grade
        if cloud_coverage == 0:
            cloudiness_groups["no clouds"].append(path_to_add)
        elif cloud_coverage == 100:
            cloudiness_groups["only clouds"].append(path_to_add)
        elif cloud_coverage < 30:
            cloudiness_groups["low"].append(path_to_add)
        elif 30 <= cloud_coverage <= 70:
            cloudiness_groups["middle"].append(path_to_add)
        elif cloud_coverage > 70:
            cloudiness_groups["high"].append(path_to_add)

        print(f"Processed {idx + 1}/{total_files} files ({((idx + 1) / total_files) * 100:.2f}%)")

    # Save the groups to a pickle file
    with open(output_file, "wb") as f:
        pickle.dump(cloudiness_groups, f)

    return cloudiness_groups


def combine_channels(mask, dataset_name):
    """Combines mask channels based on the dataset type."""
    combined = np.zeros((mask.shape[0], mask.shape[1], 3))
    if dataset_name == 'Set_2':
        combined[:, :, 0] = mask[:, :, 0]
        combined[:, :, 1] = mask[:, :, 1]
        combined[:, :, 2] = np.maximum(mask[:, :, 2], mask[:, :, 3])
    elif dataset_name == 'Biome':
        combined[:, :, 0] = mask[:, :, 0]
        combined[:, :, 1] = np.maximum(mask[:, :, 1], mask[:, :, 2])
        combined[:, :, 2] = np.maximum(mask[:, :, 3], mask[:, :, 4])
    elif dataset_name == 'Sentinel_2':
        combined[:, :, 0] = mask[:, :, 0]
        combined[:, :, 1] = mask[:, :, 2]  # Swap channels 2 and 3
        combined[:, :, 2] = mask[:, :, 1]
    return combined


def load_cloudiness_group(pickle_file, group_name):
    """
    Load a specific group of image paths from a pickle file.

    Parameters:
        pickle_file (str): Path to the pickle file containing cloudiness groups.
        group_name (str): Name of the cloudiness group to load. Possible values:
                          'low', 'middle', 'high', 'only clouds', 'no clouds'.

    Returns:
        list: List of paths corresponding to the selected cloudiness group.
    """
    # Load the cloudiness groups from the pickle file
    with open(pickle_file, "rb") as f:
        cloudiness_groups = pickle.load(f)

    # Ensure the group name exists
    if group_name not in cloudiness_groups:
        raise ValueError(f"Invalid group name '{group_name}'. Must be one of {list(cloudiness_groups.keys())}.")

    # Return the list of paths for the specified group
    return cloudiness_groups[group_name]


def get_subfolders(folder, name):
    subfolders = []
    top_level_folders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]

    for top_folder in top_level_folders:
        print(top_folder)
        inner_folders = [os.path.join(top_folder, f) for f in os.listdir(os.path.join(folder, top_folder)) if
                         os.path.isdir(os.path.join(os.path.join(folder, top_folder), f))]
        subfolders.extend(inner_folders)

    with open(name + '.pkl', "wb") as f:
        pickle.dump(subfolders, f)
    return subfolders


def normalize_image(image, min_value=0, max_value=1, mode=1):
    """
    Normalizes the input image to a specified range based on the selected mode.

    Parameters
    ----------
    image : numpy.ndarray
        Input image array of shape (height, width, channels).
    min_value : float, optional
        Minimum value of the normalized range (default is 0).
    max_value : float, optional
        Maximum value of the normalized range (default is 1).
    mode : int, optional
        Mode of normalization:
        - 1: Normalize each channel independently (default).
        - 2: Normalize all channels together based on the global min and max.

    Returns
    -------
    numpy.ndarray
        Normalized image.
    """
    image = image.astype(np.float32)  # Ensure the image is in float32

    if mode == 1:
        # Normalize each channel independently
        normalized_image = np.zeros_like(image)  # Initialize the output array
        for channel in range(image.shape[-1]):
            channel_min = np.min(image[:, :, channel])
            channel_max = np.max(image[:, :, channel])
            if channel_max - channel_min > 1e-6:  # Avoid division by zero
                normalized_image[:, :, channel] = (
                    (image[:, :, channel] - channel_min) / (channel_max - channel_min)
                )
                # Scale to the range [min_value, max_value]
                normalized_image[:, :, channel] = (
                    normalized_image[:, :, channel] * (max_value - min_value) + min_value
                )
            else:
                # Handle case where channel_min == channel_max
                normalized_image[:, :, channel] = min_value

    elif mode == 2:
        # Normalize all channels together based on global min and max
        global_min = np.min(image)
        global_max = np.max(image)
        if global_max - global_min > 1e-6:  # Avoid division by zero
            normalized_image = (image - global_min) / (global_max - global_min)
            # Scale to the range [min_value, max_value]
            normalized_image = normalized_image * (max_value - min_value) + min_value
        else:
            # Handle case where global_min == global_max
            normalized_image = np.full_like(image, min_value)

    else:
        raise ValueError("Invalid mode. Use 1 for per-channel normalization or 2 for global normalization.")

    return normalized_image




def reorder_channels(image, dataset_name, model_name):
    if model_name in ['mfcnn', 'cxn'] and dataset_name in ['Set_2', 'Biome']:
        return image[..., [3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11]]
    elif model_name in ['mfcnn', 'cxn'] and dataset_name == 'Sentinel_2':
        return sentinel_13_to_11(image, variant=2)
    elif model_name in ['mfcnn_sentinel', 'cxn_sentinel', 'mfcnn_finetuned', 'mfcnn_finetuned_lowclouds', 'mfcnn_common'] and dataset_name in ['Set_2', 'Biome']:
        return landsat_12_to_13(image, variant=1)
    elif model_name in ['mfcnn_sentinel', 'cxn_sentinel', 'mfcnn_finetuned', 'mfcnn_finetuned_lowclouds', 'mfcnn_common'] and dataset_name == 'Sentinel_2':
        return image[..., [3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12]]


def display_images_for_group(model, model_name, dataset_name, group_name, fmask_folder=None, norm_folder=None,
                             max_objects=5, pickle_file=None, shuffle=False, display_chanel=None, predict_uncertainty=False,
                             T=15):
    """
    Display images and masks for a specific cloudiness group and compute accuracy metrics for predictions.

    Parameters:
        fmask_folder (str): Path to the Fmask folder.
        norm_folder (str): Path to the folder with images and masks.
        model (keras.Model): Loaded Keras model for prediction.
        model_name (str): Name of the model.
        dataset_name (str): Dataset type ('Set_2', 'Biome', etc.).
        group_name (str): Cloudiness group name ('low', 'middle', 'high', 'only clouds', 'no clouds').
        fmask_storage (str, optional): Path to pickled Fmask subfolders. Default is False.
        mask_storage (str, optional): Path to pickled mask subfolders. Default is False.
        num_images (int): Number of images to display. Default is 5.
        pickle_file (str, optional): Path to the pickle file with cloudiness groups. Default is None.
    """
    group_folders = adjustment_input(pickle_file, group_name, max_objects, shuffle)

    metrics = aggregate_image_metrics(
        group_folders=group_folders,
        dataset_name=dataset_name,
        norm_folder=norm_folder,
        fmask_folder=fmask_folder,
        model=model,
        model_name=model_name,
        display=True,
        display_chanel=display_chanel,
        predict_uncertainty=predict_uncertainty,
        T=T
    )

    # Display average metrics
    print(f"Average Metrics for {model_name}:")
    for metric, value in metrics["predicted"].items():
        print(f"{metric.capitalize()}: {value:.2f}")

    if metrics["fmask"]:
        print(f"\nAverage Metrics for Fmask:")
        for metric, value in metrics["fmask"].items():
            print(f"{metric.capitalize()}: {value:.2f}")


def evaluate_metrics_for_group(pickle_file, group_name, model, model_name, dataset_name, fmask_folder=None, norm_folder=None, sentinel_set=None, max_objects=5):
    """
    Calculate average metrics for a specific cloudiness group.

    Parameters:
        pickle_file (str): Path to the pickle file containing cloudiness groups.
        group_name (str): Name of the cloudiness group to evaluate ('low', 'middle', etc.).
        fmask_folder (str): Path to the Fmask folder.
        norm_folder (str): Path to the folder with images and masks.
        model (keras.Model): Loaded Keras model for prediction.
        dataset_name (str): Dataset type ('Set_2', 'Biome', etc.).
        max_objects (int, optional): Limit on the number of objects to process.

    Returns:
        dict: Dictionary with average precision, recall, accuracy, and F1 score.
    """
    group_folders = adjustment_input(pickle_file, group_name, max_objects)

    metrics = aggregate_image_metrics(
        group_folders=group_folders,
        dataset_name=dataset_name,
        norm_folder=norm_folder,
        fmask_folder=fmask_folder,
        model=model,
        model_name=model_name,
        display=False
    )

    # Display results
    print(f"Metrics for group {group_name}:")
    print(f"Metrics of {model_name}:", metrics["predicted"])
    if metrics["fmask"]:
        print("Fmask Metrics:", metrics["fmask"])

    return metrics


def evaluate_all_groups(pickle_file, output_file, model, model_name, dataset_name, fmask_folder=None, norm_folder=None, sentinel_set=None, max_objects=5):
    """
    Evaluate metrics for all groups in a cloudiness pickle file and save results to a JSON file.

    Parameters:
        pickle_file (str): Path to the pickle file containing cloudiness groups.
        fmask_folder (str): Path to the Fmask folder.
        norm_folder (str): Path to the folder with images and masks.
        model (keras.Model): Loaded Keras model for prediction.
        dataset_name (str): Dataset type ('Set_2', 'Biome', etc.).
        output_file (str): Path to save the resulting metrics as a JSON file.
        max_objects (int, optional): Maximum number of objects to evaluate per group. If None, evaluate all objects.

    Returns:
        None
    """

    clouds = ['low', 'middle', 'high', 'only clouds', 'no clouds']

    results = {}

    for group_name in clouds:
        print(f"Evaluating group: {group_name}")

        # Evaluate metrics for the group
        metrics = evaluate_metrics_for_group(
            pickle_file=pickle_file,
            group_name=group_name,
            model=model,
            model_name=model_name,
            fmask_folder=fmask_folder,
            norm_folder=norm_folder,
            sentinel_set=sentinel_set,
            dataset_name=dataset_name,
            max_objects=max_objects,
        )
        results[group_name] = metrics

        print(f"Completed evaluation for group: {group_name}")

    # Save results to JSON file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")



def crossval_alpha_for_group(pickle_file, group_name, model, model_name, dataset_name, norm_folder=None, fmask_folder=None, sentinel_set=None,
                              alpha_values=None, output_file='alpha_metrics.csv', max_objects=None):
    """
    Perform cross-validation over a specific cloudiness group to find the best alpha value.

    Parameters:
        pickle_file (str): Path to the pickle file containing cloudiness groups.
        group_name (str): Name of the cloudiness group to evaluate.
        norm_folder (str): Path to the folder with images and masks.
        model (keras.Model): Loaded Keras model for prediction.
        dataset_name (str): Dataset type ('Set_2', 'Biome', etc.).
        alpha_values (list, optional): List of alpha values to evaluate. Default is [0.5, 0.7, 0.95].
        output_file (str, optional): CSV file to save metrics. Default is 'alpha_metrics.csv'.
        max_objects (int, optional): Maximum number of objects to evaluate from the group. Default is None (all objects).

    Returns:
        dict: The best alpha value and its corresponding metrics.
    """
    if alpha_values is None:
        alpha_values = [0.5, 0.7, 0.95]

    # Load and validate group folders
    group_folders = adjustment_input(
        pickle_file=pickle_file,
        group_name=group_name,
        max_objects=max_objects
    )

    best_alpha = None
    best_metrics = {
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0
    }
    best_avg_f1 = 0

    # Initialize CSV file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Alpha', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

    for alpha in alpha_values:
        print(f"\nEvaluating for alpha = {alpha}...")

        # Aggregate metrics for the current alpha value
        metrics = aggregate_image_metrics(
            group_folders=group_folders,
            dataset_name=dataset_name,
            norm_folder=norm_folder,
            fmask_folder=fmask_folder,
            model=model,
            model_name=model_name,
            display=False,
            alpha=alpha
        )

        avg_accuracy = metrics['predicted']['accuracy']
        avg_precision = metrics['predicted']['precision']
        avg_recall = metrics['predicted']['recall']
        avg_f1 = metrics['predicted']['f1']

        # Display average metrics
        print(f"Metrics for alpha = {alpha}:")
        print(f"Accuracy: {avg_accuracy:.2f}")
        print(f"Precision: {avg_precision:.2f}")
        print(f"Recall: {avg_recall:.2f}")
        print(f"F1 Score: {avg_f1:.2f}")

        # Write metrics to CSV file
        with open(output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([alpha, avg_accuracy, avg_precision, avg_recall, avg_f1])

        # Update the best alpha if this alpha has a higher average F1 score
        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            best_alpha = alpha
            best_metrics = {
                'accuracy': avg_accuracy,
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1
            }

    # Display best metrics
    print("\nBest Alpha and Corresponding Metrics:")
    print(f"Best Alpha: {best_alpha}")
    print(f" - Accuracy: {best_metrics['accuracy']:.4f}")
    print(f" - Precision: {best_metrics['precision']:.4f}")
    print(f" - Recall: {best_metrics['recall']:.4f}")
    print(f" - F1 Score: {best_metrics['f1']:.4f}")

    return {
        'best_alpha': best_alpha,
        'best_metrics': best_metrics
    }


def analyze_brightness_contrast(
    pickle_file, model, dataset_name, model_name, norm_folder=None, max_objects=5, shuffle=False, shuffle_seed=None, selected_groups=None, output_file="results.json"
):
    """
    Analysis of brightness, contrast, and additional metrics in the predicted masks of the second channel.

    Parameters:
        - pickle_file: str
            Path to a pickle file containing cloud groups ('no clouds', 'only clouds', 'low', 'middle', 'high').
            The pickle file should contain a dictionary where keys are group names and values are lists of
            tuples (path to image, path to mask).
        - model: object
            Pre-trained model used for generating predictions.
        - dataset_name: str
            Name of the dataset used for preprocessing.
        - model_name: str
            Name of the model used for predictions.
        - max_objects: int, optional
            Number of images to analyze from each group. If None, all images are used.
        - shuffle: bool, optional
            Whether to shuffle the images before selecting samples. Default is False.
        - shuffle_seed: int, optional
            Seed for shuffling, ensuring reproducibility. Default is None.
        - selected_groups: list, optional
            List of cloud groups to analyze. If None, all groups are analyzed.
        - output_file: str, optional
            Path to the file where results will be saved. Default is 'results.json'.

    Returns:
        dict: Statistics of brightness, contrast, and additional metrics for each cloud group.
    """
    if not pickle_file:
        raise ValueError("A pickle file with cloudiness groups must be provided.")

    # Load data from the pickle file
    try:
        with open(pickle_file, "rb") as f:
            cloudiness_groups = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Error loading pickle file: {e}")

    # Filter groups if selected_groups is provided
    if selected_groups:
        cloudiness_groups = {k: v for k, v in cloudiness_groups.items() if k in selected_groups}

    if not cloudiness_groups:
        raise ValueError("No valid cloud groups found to analyze.")

    results = {}

    for cloud_group, image_mask_list in cloudiness_groups.items():
        print(f"Analyzing group: {cloud_group}...")

        if shuffle:
            if shuffle_seed is not None:
                random.seed(shuffle_seed)
            random.shuffle(image_mask_list)

        # Select the desired number of samples
        if max_objects is not None:
            image_mask_list = image_mask_list[:max_objects]

        print(f"Number of images to process: {len(image_mask_list)}")

        brightness_values = []  # Brightness of the pixels in the second channel
        contrast_values = []  # Difference between max and min values of the second channel
        low_intensity_fractions = []  # Fraction of pixels below a low-intensity threshold
        local_uniformities = []  # Local uniformity values
        high_intensity_fractions = []  # Fraction of pixels above a high-intensity threshold
        blur_indices = []  # Blur indices for the mask
        entropies = []  # Entropy of probability distribution
        mid_intensity_fractions = []  # Fraction of mid-intensity pixels
        skewness_values = []  # Skewness of the distribution
        kurtosis_values = []  # Kurtosis of the distribution
        normalized_entropies = []  # Normalized entropy values
        orderliness_coefficients = []  # Coefficients of orderliness

        for idx, folder in enumerate(image_mask_list):
            print(f"Processing image {idx + 1}/{len(image_mask_list)} in group {cloud_group}")
            if dataset_name in ['Set_2', 'Biome']:
                folder2_path = os.path.join(norm_folder, folder)

                # Paths to the images and masks
                image_path = os.path.join(folder2_path, 'image.npy')
                mask_path = os.path.join(folder2_path, 'mask.npy')

                if not (os.path.exists(image_path) and os.path.exists(mask_path)):
                    continue
            else:
                image_path, mask_path = folder
                fmask_path = None
                if not (os.path.exists(image_path) and os.path.exists(mask_path)):
                    continue

            # Load the image and preprocess
            image, mask, binary_fmask = load_and_preprocess(image_path, mask_path, None, dataset_name, model_name)

            # Get the model prediction
            predicted_mask = model.predict(image)[0]  # Remove batch dimension

            # Extract the second channel (clouds)
            cloud_channel = predicted_mask[:, :, 1]

            # Brightness statistics
            brightness_values.extend(cloud_channel.flatten())  # Collect all pixel values

            # Contrast statistics
            max_value = np.max(cloud_channel)
            min_value = np.min(cloud_channel)
            contrast = max_value - min_value
            contrast_values.append(contrast)

            # Low-intensity fraction
            low_threshold = 0.2
            low_intensity_fraction = np.mean(cloud_channel < low_threshold)
            low_intensity_fractions.append(low_intensity_fraction)

            # Local uniformity (variance of small patches)
            patch_size = 8
            local_variance = []
            for i in range(0, cloud_channel.shape[0], patch_size):
                for j in range(0, cloud_channel.shape[1], patch_size):
                    patch = cloud_channel[i:i + patch_size, j:j + patch_size]
                    if patch.size > 0:
                        local_variance.append(np.var(patch))
            local_uniformities.append(np.mean(local_variance))

            # High-intensity fraction
            high_threshold = 0.7
            high_intensity_fraction = np.mean(cloud_channel > high_threshold)
            high_intensity_fractions.append(high_intensity_fraction)

            # Blur index (variance of Laplacian)
            laplacian = cv2.Laplacian(cloud_channel.astype(np.float64), cv2.CV_64F)
            blur_index = np.var(laplacian)
            blur_indices.append(blur_index)

            # Entropy of probabilities
            cloud_channel_flat = cloud_channel.flatten()
            entropy = -np.sum(cloud_channel_flat * np.log(cloud_channel_flat + 1e-10)) / len(cloud_channel_flat)
            entropies.append(entropy)

            # Normalized entropy
            max_entropy = np.log(len(cloud_channel_flat))
            normalized_entropy = entropy / max_entropy
            normalized_entropies.append(normalized_entropy)

            # Mid-intensity fraction
            mid_intensity_fraction = np.mean((cloud_channel >= 0.2) & (cloud_channel <= 0.8))
            mid_intensity_fractions.append(mid_intensity_fraction)

            # Skewness
            skewness = np.mean(((cloud_channel_flat - np.mean(cloud_channel_flat)) / np.std(cloud_channel_flat))**3)
            skewness_values.append(skewness)

            # Kurtosis
            kurtosis = np.mean(((cloud_channel_flat - np.mean(cloud_channel_flat)) / np.std(cloud_channel_flat))**4) - 3
            kurtosis_values.append(kurtosis)

            # Orderliness coefficient
            diff_x = np.abs(np.diff(cloud_channel, axis=0))  # Разность вдоль оси X
            diff_y = np.abs(np.diff(cloud_channel, axis=1))  # Разность вдоль оси Y

            orderliness_coefficient = np.mean(diff_x[:, :-1] + diff_y[:-1, :])
            orderliness_coefficients.append(orderliness_coefficient)

        # Calculate final statistics for the current cloud group
        results[cloud_group] = {
            'mean_brightness': float(np.mean(brightness_values)) if brightness_values else None,
            'std_brightness': float(np.std(brightness_values)) if brightness_values else None,
            'mean_contrast': float(np.mean(contrast_values)) if contrast_values else None,
            'std_contrast': float(np.std(contrast_values)) if contrast_values else None,
            'mean_low_intensity_fraction': float(np.mean(low_intensity_fractions)) if low_intensity_fractions else None,
            'mean_local_uniformity': float(np.mean(local_uniformities)) if local_uniformities else None,
            'mean_high_intensity_fraction': float(np.mean(high_intensity_fractions)) if high_intensity_fractions else None,
            'mean_blur_index': float(np.mean(blur_indices)) if blur_indices else None,
            'mean_entropy': float(np.mean(entropies)) if entropies else None,
            'mean_normalized_entropy': float(np.mean(normalized_entropies)) if normalized_entropies else None,
            'mean_mid_intensity_fraction': float(np.mean(mid_intensity_fractions)) if mid_intensity_fractions else None,
            'mean_skewness': float(np.mean(skewness_values)) if skewness_values else None,
            'mean_kurtosis': float(np.mean(kurtosis_values)) if kurtosis_values else None,
            'mean_orderliness_coefficient': float(np.mean(orderliness_coefficients)) if orderliness_coefficients else None,
            'num_images': len(image_mask_list),
        }

    # Save results to the output file
    try:
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_file}")
    except Exception as e:
        raise ValueError(f"Error saving results to file: {e}")

    return results



def classify_no_clouds(predicted_mask, binary_mask, model_name, using=False):
    """
    Classifies a predicted mask as "no clouds" and modifies it if criteria are met.

    Parameters:
        - predicted_mask: np.ndarray
            Predicted mask with shape (H, W, C), where the second channel represents cloud predictions.
        - model_name: str
            Name of the model used for predictions.
        - using: bool
            Whether to apply the classification logic. Default is False.

    Returns:
        - np.ndarray
            Modified mask with zeros if classified as "no clouds", otherwise the original mask.
    """
    if using == False:
        return binary_mask

    if not np.all(binary_mask == 1):
        return binary_mask

    if predicted_mask.shape[-1] < 3:
        raise ValueError("The predicted mask must have at least 3 channels.")

    # Extract the second channel (clouds)
    if model_name in ['cxn', 'mfcnn']:
        cloud_channel = predicted_mask[:, :, -1]
    elif model_name in ['cxn_sentinel', 'mfcnn_sentinel', 'mfcnn_finetuned', 'mfcnn_finetuned_lowclouds', 'mfcnn_common']:
        cloud_channel = predicted_mask[:, :, -2]

    # Calculate metrics
    brightness = np.mean(cloud_channel)
    low_threshold = 0.2
    high_threshold = 0.8

    low_intensity_fraction = np.mean(cloud_channel < low_threshold)
    high_intensity_fraction = np.mean(cloud_channel > high_threshold)

    # Skewness
    cloud_channel_flat = cloud_channel.flatten()
    skewness = np.mean(((cloud_channel_flat - np.mean(cloud_channel_flat)) / np.std(cloud_channel_flat)) ** 3)

    # Kurtosis
    kurtosis = np.mean(((cloud_channel_flat - np.mean(cloud_channel_flat)) / np.std(cloud_channel_flat)) ** 4) - 3

    # Debugging outputs
    print("Brightness:", brightness)
    print("Low Intensity Fraction:", low_intensity_fraction)
    print("High Intensity Fraction:", high_intensity_fraction)
    print("Skewness:", skewness)
    print("Kurtosis:", kurtosis)

    # Classification criteria for "no clouds" vs "only clouds"
    if (
            brightness < 0.63 and
            low_intensity_fraction < 0.05 and
            high_intensity_fraction < 0.4 and
            skewness > -1 and
            kurtosis < 9
    ):
        print('Classified as "no clouds"')
        binary_mask[:, :] = 0  # Set all values to 0 if classified as "no clouds"

    return binary_mask


def predict_with_uncertainty(model, image, T=100):
    """
    Perform multiple forward passes with active MC-Dropout to estimate uncertainty.

    Parameters:
    - f_model: The model with MC-Dropout enabled.
    - image: Input data (e.g., images or batches).
    - T: Number of forward passes to perform.

    Returns:
    - mean_pred: Mean prediction across all passes.
    - std_pred: Standard deviation of predictions (uncertainty).
    """

    f_model = MCDropoutModel(model)
    preds = np.array([f_model(image, training=True).numpy() for _ in range(T)])
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)

    return mean_pred, std_pred





