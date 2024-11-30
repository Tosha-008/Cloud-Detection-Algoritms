import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from MFCNN.model_mfcnn_def import *
import pickle
from output.defs_for_output import calculate_metrics, compute_average_metrics
import json
import csv
import cv2

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

def split_images_by_cloudiness(folder, output_file, dataset_name='Set_2', mask_storage=None):
    """
    Function to iterate over all images in a folder, calculate their cloudiness levels,
    and split them into separate lists by cloudiness grade.

    Parameters:
        folder (str): Path to the folder containing subfolders with images and masks.
        output_file (str): Path to save the resulting cloudiness lists as a pickle file.
        dataset_name (str): Dataset type, e.g., 'Set_2' or 'Biome'.

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

    # Iterate through all subfolders
    if mask_storage:
        with open(mask_storage, "rb") as f:
            norm_subfolders = pickle.load(f)

    if mask_storage == None:
        norm_subfolders = get_subfolders(folder, 'norm_subfolders')

    total_subfolders = len(norm_subfolders)
    for idx, path in enumerate(norm_subfolders):
        # Paths to the image and mask
        subfolder = os.path.join(folder, path)
        mask_path = os.path.join(subfolder, 'mask.npy')

        if not os.path.exists(mask_path):
            continue

        # Load the mask
        mask = np.load(mask_path)

        # Combine channels based on dataset type
        mask_combined = np.zeros((mask.shape[0], mask.shape[1], 3))

        if dataset_name == 'Set_2':
            mask_combined[:, :, 0] = mask[:, :, 0]
            mask_combined[:, :, 1] = mask[:, :, 1]
            mask_combined[:, :, 2] = np.maximum(mask[:, :, 2], mask[:, :, 3])
        elif dataset_name == 'Biome':
            mask_combined[:, :, 0] = mask[:, :, 0]
            mask_combined[:, :, 1] = np.maximum(mask[:, :, 1], mask[:, :, 2])
            mask_combined[:, :, 2] = np.maximum(mask[:, :, 3], mask[:, :, 4])

        # Calculate the cloudiness level
        cloud_pixels = np.sum(mask_combined[:, :, -1] > 0)  # Assuming cloud pixels are marked with values > 0
        total_pixels = mask_combined[:, :, -1].size
        cloud_coverage = (cloud_pixels / total_pixels) * 100

        # Categorize the image by cloudiness grade
        if cloud_coverage == 0:
            cloudiness_groups["no clouds"].append(path)
        elif cloud_coverage == 100:
            cloudiness_groups["only clouds"].append(path)
        elif cloud_coverage < 30:
            cloudiness_groups["low"].append(path)
        elif 30 <= cloud_coverage <= 70:
            cloudiness_groups["middle"].append(path)
        elif cloud_coverage > 70:
            cloudiness_groups["high"].append(path)

        print(f"Processed {idx}/{total_subfolders} subfolders ({(idx / total_subfolders) * 100:.2f}%)")

    # Save the groups to a pickle file
    with open(output_file, "wb") as f:
        pickle.dump(cloudiness_groups, f)

    return cloudiness_groups


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

def display_images_for_group(fmask_folder, norm_folder, model, model_name, dataset_name, group_name,
                             num_images=5, pickle_file=None):
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

    # Get the list of folders from both directories

    # Load cloudiness groups from pickle file
    if pickle_file:
        with open(pickle_file, "rb") as f:
            cloudiness_groups = pickle.load(f)

        if group_name not in cloudiness_groups:
            raise ValueError(f"Group '{group_name}' not found in pickle file.")

        group_folders = cloudiness_groups[group_name]
    else:
        raise ValueError("A pickle file with cloudiness groups must be provided.")

    count = 0  # Counter for the number of displayed images

    all_precision_pred = []
    all_recall_pred = []
    all_accuracy_pred = []
    all_f1_pred = []

    all_precision_fmask = []
    all_recall_fmask = []
    all_accuracy_fmask = []
    all_f1_fmask = []

    # Loop through the folders in the specified group
    for folder in group_folders:
        folder2_path = os.path.join(norm_folder, folder)
        folder1_name = folder[:-3] + "_" + folder[-3:]
        folder1_path = os.path.join(fmask_folder, folder1_name)

        # Paths to the images and masks
        image_path = os.path.join(folder2_path, 'image.npy')
        mask_path = os.path.join(folder2_path, 'mask.npy')
        fmask_path = os.path.join(folder1_path, 'image.npy')

        if not (os.path.exists(image_path) and os.path.exists(mask_path) and os.path.exists(fmask_path)):
            continue

        # Load the images and masks
        image = np.load(image_path)
        mask = np.load(mask_path)
        fmask = np.load(fmask_path)
        binary_fmask = np.where(fmask == 2, 1, 0)

        # Combine channels for the mask
        mask_combined = np.zeros((mask.shape[0], mask.shape[1], 3))

        if dataset_name == 'Set_2':
            mask_combined[:, :, 0] = mask[:, :, 0]
            mask_combined[:, :, 1] = mask[:, :, 1]
            mask_combined[:, :, 2] = np.maximum(mask[:, :, 2], mask[:, :, 3])
        elif dataset_name == 'Biome':
            mask_combined[:, :, 0] = mask[:, :, 0]
            mask_combined[:, :, 1] = np.maximum(mask[:, :, 1], mask[:, :, 2])
            mask_combined[:, :, 2] = np.maximum(mask[:, :, 3], mask[:, :, 4])

        # Prepare the image for prediction
        desired_order = [3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11]
        image_reordered = image[..., desired_order]
        image_reordered_expanded = np.expand_dims(image_reordered, axis=0)
        pred_mask = model.predict(image_reordered_expanded)

        # Apply threshold based on alpha and model_name
        alpha = 0.5 if model_name == 'cloudfcn' else 0.05 if model_name == 'mfcnn' else 0.05
        pred_mask_binary = (pred_mask.squeeze()[:, :, -1] > alpha).astype(float)

        if model_name == 'cxn':
            pred_mask_binary = (pred_mask.squeeze()[:, :, -1]  > alpha).astype(float)
            kernel = np.ones((3, 3), np.uint8)
            pred_mask_binary = cv2.morphologyEx(pred_mask_binary, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(pred_mask_binary.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            output_mask = np.zeros_like(pred_mask_binary, dtype=np.uint8)
            min_area = 5
            for contour in contours:
                if cv2.contourArea(contour) > min_area:
                    cv2.drawContours(output_mask, [contour], -1, 1, -1)

            kernel = np.ones((3, 3), np.uint8)
            dilated_mask = cv2.dilate(output_mask, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            final_mask = np.zeros_like(output_mask)
            cv2.drawContours(final_mask, contours, -1, 1, -1)

            pred_mask_binary = final_mask
        # Normalize the image
        image = image.astype(np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        # Calculate metrics
        mask_flat = mask[:, :, -1].flatten()
        binary_fmask_flat = binary_fmask.flatten()

        accuracy_pred = precision_pred = recall_pred = f1_pred = 0
        if dataset_name in ['Set_2', 'Biome']:
            accuracy_pred, precision_pred, recall_pred, f1_pred = calculate_metrics(
                mask[:, :, -1].squeeze(), pred_mask_binary
            )

        accuracy_fmask = accuracy_score(mask_flat, binary_fmask_flat)
        precision_fmask = precision_score(mask_flat, binary_fmask_flat, zero_division=1)
        recall_fmask = recall_score(mask_flat, binary_fmask_flat, zero_division=1)
        f1_fmask = f1_score(mask_flat, binary_fmask_flat, zero_division=1)

        # Store the metrics
        all_accuracy_pred.append(accuracy_pred)
        all_precision_pred.append(precision_pred)
        all_recall_pred.append(recall_pred)
        all_f1_pred.append(f1_pred)

        all_accuracy_fmask.append(accuracy_fmask)
        all_precision_fmask.append(precision_fmask)
        all_recall_fmask.append(recall_fmask)
        all_f1_fmask.append(f1_fmask)

        # Display the images
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))  # Create 2x2 grid

        image_rgb = image[:, :, :3]
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title(f"Image from {folder1_name}")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(mask[:, :, -1], cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title("Mask")
        axes[0, 1].axis('off')

        axes[1, 0].imshow(binary_fmask, cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title("Fmask")
        axes[1, 0].axis('off')

        # Add metrics below the corresponding images
        metrics_text_fmask = (
            f'Accuracy: {accuracy_fmask:.2f}\n'
            f'Precision: {precision_fmask:.2f}\n'
            f'Recall: {recall_fmask:.2f}\n'
            f'F1 Score: {f1_fmask:.2f}'
        )
        axes[1, 0].text(0.5, -0.2, metrics_text_fmask, color='black', ha='center', va='top',
                        transform=axes[1, 0].transAxes, fontsize=10)

        axes[1, 1].imshow(pred_mask_binary, cmap='gray', vmin=0, vmax=1)
        axes[1, 1].set_title(f'Predicted Mask')
        axes[1, 1].axis('off')

        # Add metrics below the corresponding images
        metrics_text_pred = (
            f'Accuracy: {accuracy_pred:.2f}\n'
            f'Precision: {precision_pred:.2f}\n'
            f'Recall: {recall_pred:.2f}\n'
            f'F1 Score: {f1_pred:.2f}'
        )
        axes[1, 1].text(0.5, -0.2, metrics_text_pred, color='black', ha='center', va='top',
                        transform=axes[1, 1].transAxes, fontsize=10)

        plt.tight_layout()
        plt.show()

        # Increment the counter
        count += 1

        # Break the loop if the required number of images has been displayed
        if count >= num_images:
            break

    # Calculate and print the average metrics for the entire batch
    avg_accuracy_pred = np.mean(all_accuracy_pred)
    avg_precision_pred = np.mean(all_precision_pred)
    avg_recall_pred = np.mean(all_recall_pred)
    avg_f1_pred = np.mean(all_f1_pred)

    avg_accuracy_fmask = np.mean(all_accuracy_fmask)
    avg_precision_fmask = np.mean(all_precision_fmask)
    avg_recall_fmask = np.mean(all_recall_fmask)
    avg_f1_fmask = np.mean(all_f1_fmask)

    print(f'Average Accuracy of Model {model_name}: {avg_accuracy_pred:.2f}')
    print(f'Average Precision of Model {model_name}: {avg_precision_pred:.2f}')
    print(f'Average Recall of Model {model_name}: {avg_recall_pred:.2f}')
    print(f'Average F1 Score of Model {model_name}: {avg_f1_pred:.2f}')

    print(f'Average Accuracy of Fmask: {avg_accuracy_fmask:.2f}')
    print(f'Average Precision of Fmask: {avg_precision_fmask:.2f}')
    print(f'Average Recall of Fmask: {avg_recall_fmask:.2f}')
    print(f'Average F1 Score of Fmask: {avg_f1_fmask:.2f}')


def evaluate_metrics_for_group(pickle_file, group_name, fmask_folder, norm_folder, model, dataset_name, max_objects=None):
    """
    Calculate average metrics for a specific cloudiness group.

    Parameters:
        pickle_file (str): Path to the pickle file containing cloudiness groups.
        group_name (str): Name of the cloudiness group to evaluate ('low', 'middle', etc.).
        fmask_folder (str): Path to the Fmask folder.
        norm_folder (str): Path to the folder with images and masks.
        model (keras.Model): Loaded Keras model for prediction.
        dataset_name (str): Dataset type ('Set_2', 'Biome', etc.).

    Returns:
        dict: Dictionary with average precision, recall, accuracy, and F1 score.
    """
    # Load the cloudiness groups from the pickle file
    with open(pickle_file, "rb") as f:
        cloudiness_groups = pickle.load(f)

    if group_name not in cloudiness_groups:
        raise ValueError(f"Group '{group_name}' not found in pickle file.")

    # Get the list of paths for the specified group
    paths = cloudiness_groups[group_name]

    if max_objects is not None:
        random.shuffle(paths)
        paths = paths[:max_objects]

    # Initialize metrics accumulators
    all_precision_pred = []
    all_recall_pred = []
    all_accuracy_pred = []
    all_f1_pred = []

    all_precision_fmask = []
    all_recall_fmask = []
    all_accuracy_fmask = []
    all_f1_fmask = []

    total_paths = len(paths)

    for idx, folder in enumerate(paths, start=1):
        folder2_path = os.path.join(norm_folder, folder)
        folder1_name = folder[:-3] + "_" + folder[-3:]  # Modify to match Fmask format
        folder1_path = os.path.join(fmask_folder, folder1_name)

        # Paths to the images and masks
        image_path = os.path.join(folder2_path, 'image.npy')
        mask_path = os.path.join(folder2_path, 'mask.npy')
        fmask_path = os.path.join(folder1_path, 'image.npy')

        if not (os.path.exists(image_path) and os.path.exists(mask_path) and os.path.exists(fmask_path)):
            continue

        # Load the images
        image = np.load(image_path)
        mask = np.load(mask_path)
        fmask = np.load(fmask_path)
        binary_fmask = np.where(fmask == 2, 1, 0)

        # Combine channels for the mask
        mask_combined = np.zeros((mask.shape[0], mask.shape[1], 3))

        if dataset_name == 'Set_2':
            mask_combined[:, :, 0] = mask[:, :, 0]
            mask_combined[:, :, 1] = mask[:, :, 1]
            mask_combined[:, :, 2] = np.maximum(mask[:, :, 2], mask[:, :, 3])
        elif dataset_name == 'Biome':
            mask_combined[:, :, 0] = mask[:, :, 0]
            mask_combined[:, :, 1] = np.maximum(mask[:, :, 1], mask[:, :, 2])
            mask_combined[:, :, 2] = np.maximum(mask[:, :, 3], mask[:, :, 4])

        desired_order = [3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11]
        image_reordered = image[..., desired_order]
        image_reordered_expanded = np.expand_dims(image_reordered, axis=0)
        pred_mask = model.predict(image_reordered_expanded)

        # Apply threshold based on alpha
        alpha = 0.35  # Adjust based on model specifics
        pred_mask_binary = (pred_mask.squeeze()[:, :, -1] > alpha).astype(float)

        # Normalize the image
        image = image.astype(np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        # Calculate metrics for the predicted mask
        if dataset_name == 'Biome':
            accuracy_pred, precision_pred, recall_pred, f1_pred = calculate_metrics(mask[:, :, -1].squeeze(), pred_mask_binary)
        elif dataset_name == 'Set_2':
            accuracy_pred, precision_pred, recall_pred, f1_pred = calculate_metrics(mask[:, :, -1].squeeze(), pred_mask_binary)
        elif dataset_name == 'Sentinel_2':
            accuracy_pred, precision_pred, recall_pred, f1_pred = calculate_metrics(mask[:, :, -2].squeeze(), pred_mask_binary)

        # Flatten arrays for Fmask comparison
        mask_flat = mask[:, :, -1].flatten()
        binary_fmask_flat = binary_fmask.flatten()

        precision_fmask = precision_score(mask_flat, binary_fmask_flat, zero_division=1)
        recall_fmask = recall_score(mask_flat, binary_fmask_flat, zero_division=1)
        f1_fmask = f1_score(mask_flat, binary_fmask_flat, zero_division=1)
        accuracy_fmask = accuracy_score(mask_flat, binary_fmask_flat)

        # Store the metrics
        all_accuracy_pred.append(accuracy_pred)
        all_precision_pred.append(precision_pred)
        all_recall_pred.append(recall_pred)
        all_f1_pred.append(f1_pred)

        all_accuracy_fmask.append(accuracy_fmask)
        all_precision_fmask.append(precision_fmask)
        all_recall_fmask.append(recall_fmask)
        all_f1_fmask.append(f1_fmask)

        print(f"Processed {idx}/{total_paths} folders ({(idx / total_paths) * 100:.2f}%)")
    # Calculate and return average metrics
    return {
        'model': {
            'accuracy': np.mean(all_accuracy_pred),
            'precision': np.mean(all_precision_pred),
            'recall': np.mean(all_recall_pred),
            'f1_score': np.mean(all_f1_pred)
        },
        'fmask': {
            'accuracy': np.mean(all_accuracy_fmask),
            'precision': np.mean(all_precision_fmask),
            'recall': np.mean(all_recall_fmask),
            'f1_score': np.mean(all_f1_fmask)
        }
    }

def evaluate_all_groups(pickle_file, fmask_folder, norm_folder, model, dataset_name, output_file, max_objects=None):
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


    with open(pickle_file, "rb") as f:
        cloudiness_groups = pickle.load(f)

    results = {}

    for group_name in cloudiness_groups.keys():
        print(f"Evaluating group: {group_name}")
        metrics = evaluate_metrics_for_group(
            pickle_file, group_name, fmask_folder, norm_folder, model, dataset_name, max_objects
        )
        results[group_name] = metrics

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")


def crossval_alpha_for_group(pickle_file, group_name, fmask_folder, norm_folder, model, dataset_name,
                              alpha_values=None, output_file='alpha_metrics.csv', max_objects=None):
    """
    Perform cross-validation over a specific cloudiness group to find the best alpha value.

    Parameters:
        pickle_file (str): Path to the pickle file containing cloudiness groups.
        group_name (str): Name of the cloudiness group to evaluate.
        fmask_folder (str): Path to the Fmask folder.
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

    # Load group paths from pickle file
    with open(pickle_file, "rb") as f:
        cloudiness_groups = pickle.load(f)

    if group_name not in cloudiness_groups:
        raise ValueError(f"Group '{group_name}' not found in pickle file.")

    paths = cloudiness_groups[group_name]
    if max_objects:
        paths = paths[:max_objects]

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

        all_accuracies = []
        all_precisions = []
        all_recalls = []
        all_f1s = []

        for idx, folder in enumerate(paths):
            folder2_path = os.path.join(norm_folder, folder)
            folder1_name = folder[:-3] + "_" + folder[-3:]  # Modify to match Fmask format
            folder1_path = os.path.join(fmask_folder, folder1_name)

            # Paths to the images and masks
            image_path = os.path.join(folder2_path, 'image.npy')
            mask_path = os.path.join(folder2_path, 'mask.npy')
            fmask_path = os.path.join(folder1_path, 'image.npy')

            if not (os.path.exists(image_path) and os.path.exists(mask_path) and os.path.exists(fmask_path)):
                continue

            # Load the images and masks
            image = np.load(image_path)
            mask = np.load(mask_path)
            predictions = model.predict(np.expand_dims(image, axis=0)).squeeze()

            # Apply threshold based on alpha
            pred_mask_binary = (predictions[:, :, -1] > alpha).astype(float)

            if dataset_name == "Biome":
                accuracy, precision, recall, f1 = calculate_metrics(mask[:, :, -1].squeeze(), pred_mask_binary)
            elif dataset_name == "Set_2":
                accuracy, precision, recall, f1 = calculate_metrics(mask[:, :, -1].squeeze(), pred_mask_binary)
            elif dataset_name == "Sentinel_2":
                accuracy, precision, recall, f1 = calculate_metrics(mask[:, :, -2].squeeze(), pred_mask_binary)

            all_accuracies.append(accuracy)
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)

            print(f"Processed {idx + 1}/{len(paths)} images for alpha = {alpha}...")

        # Compute the average metrics for the current alpha
        avg_accuracy, avg_precision, avg_recall, avg_f1 = compute_average_metrics(
            all_accuracies, all_precisions, all_recalls, all_f1s
        )

        print(f"Metrics for alpha = {alpha}:")
        print(f" - Accuracy: {avg_accuracy:.4f}")
        print(f" - Precision: {avg_precision:.4f}")
        print(f" - Recall: {avg_recall:.4f}")
        print(f" - F1 Score: {avg_f1:.4f}")

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