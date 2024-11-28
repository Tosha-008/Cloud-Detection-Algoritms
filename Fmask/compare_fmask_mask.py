import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from tensorflow.keras.models import load_model
from MFCNN.model_mfcnn_def import *
import pickle

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


def get_subfolders(folder):
    subfolders = []
    top_level_folders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]

    for top_folder in top_level_folders:
        print(top_folder)
        inner_folders = [os.path.join(top_folder, f) for f in os.listdir(os.path.join(folder, top_folder)) if
                         os.path.isdir(os.path.join(os.path.join(folder, top_folder), f))]
        subfolders.extend(inner_folders)

    with open("subfolders.pkl", "wb") as f:
        pickle.dump(subfolders, f)
    return subfolders

def display_images(fmask_folder, norm_folder,
                   fmask_storage = False,
                   mask_storage = False,
                   num_images=5):
    """
    Function to display images and masks, and compute accuracy metrics for predictions.

    :param fmask_folder: Path to the Fmask folder
    :param norm_folder: Path to the folder with images and masks
    :param num_images: Number of images to display
    """
    # Get the list of folders from both directories

    if fmask_storage and mask_storage:
        with open("subfolders.pkl", "rb") as f:
            fmask_subfolders = pickle.load(f)
        with open("subfolders_1.pkl", "rb") as f:
            norm_subfolders = pickle.load(f)

    if not fmask_storage:
        fmask_subfolders = get_subfolders(fmask_folder)
        norm_subfolders = get_subfolders(norm_folder)

    # Counter for the number of displayed images
    count = 0

    # Shuffle the folder lists for random selection
    random.shuffle(fmask_subfolders)
    random.shuffle(norm_subfolders)

    # Initialize lists to store metrics
    all_precision_pred = []
    all_recall_pred = []
    all_accuracy_pred = []
    all_f1_pred = []

    all_precision_fmask = []
    all_recall_fmask = []
    all_accuracy_fmask = []
    all_f1_fmask = []

    # Search for matching folders and display images
    for folder in norm_subfolders:
        folder2_name = folder
        folder1_name = folder2_name[:3] + "_" + folder2_name[3:]  # Modify to match folder1 format
        if folder1_name in fmask_subfolders:
            folder1_path = os.path.join(fmask_folder, folder1_name)
            folder2_path = os.path.join(norm_folder, folder2_name)

            # Paths to the images and masks
            image_path = os.path.join(folder2_path, 'image.npy')
            mask_path = os.path.join(folder2_path, 'mask.npy')
            fmask_path = os.path.join(folder1_path, 'image.npy')

            # Load the images
            image = np.load(image_path)
            mask = np.load(mask_path)
            fmask = np.load(fmask_path)
            binary_fmask = np.where(fmask == 2, 1, 0)

            desired_order = [3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11]
            image_reordered = image[..., desired_order]
            image_reordered_expanded = np.expand_dims(image_reordered, axis=0)
            prediction = model.predict(image_reordered_expanded)

            alpha = 0.1
            pred_mask_binary = (prediction.squeeze()[:, :, -1] > alpha).astype(float)

            # Normalize the image (assuming image data is in the range 0-255)
            image = image.astype(np.float32)
            image = (image - np.min(image)) / (np.max(image) - np.min(image))

            # Combine channels 1 & 2 and 3 & 4 of the mask
            mask_combined = np.zeros((mask.shape[0], mask.shape[1], 3))  # New mask with 3 channels
            mask_combined[:, :, 0] = mask[:, :, 0]
            mask_combined[:, :, 1] = np.maximum(mask[:, :, 1], mask[:, :, 2])
            mask_combined[:, :, 2] = np.maximum(mask[:, :, 3], mask[:, :, 4])

            cmap = ListedColormap(['black', 'blue', 'green', 'yellow', 'orange', 'red'])

            cmap_channel_1 = ListedColormap(['black', 'blue'])
            cmap_channel_2 = ListedColormap(['black', 'green'])
            cmap_channel_3 = ListedColormap(['black', 'red'])

            # Calculate the metrics
            pred_mask_flat = pred_mask_binary.flatten()
            mask_flat = mask[:, :, -1].flatten()
            binary_fmask_flat = binary_fmask.flatten()

            precision_pred = precision_score(mask_flat, pred_mask_flat, zero_division=1)
            recall_pred = recall_score(mask_flat, pred_mask_flat, zero_division=1)
            f1_pred = f1_score(mask_flat, pred_mask_flat, zero_division=1)
            accuracy_pred = accuracy_score(mask_flat, pred_mask_flat)

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

            # Display the images
            fig, axes = plt.subplots(2, 2, figsize=(18, 12))  # Create 2x2 grid

            image_rgb = image[:, :, :3]
            axes[0, 0].imshow(image_rgb)
            axes[0, 0].set_title(f"Image from {folder1_name}")
            axes[0, 0].axis('off')

            axes[0, 1].imshow(mask[:, :, -1], cmap='gray', vmin=0, vmax=1)  # Channel 2
            axes[0, 1].set_title(f"Mask from {folder1_name}")
            axes[0, 1].axis('off')

            axes[1, 0].imshow(binary_fmask, cmap='gray', vmin=0, vmax=1)
            axes[1, 0].set_title(f"Fmask from {folder2_name}")
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

    print(f'Average Accuracy Pred: {avg_accuracy_pred:.2f}')
    print(f'Average Precision Pred: {avg_precision_pred:.2f}')
    print(f'Average Recall Pred: {avg_recall_pred:.2f}')
    print(f'Average F1 Score Pred: {avg_f1_pred:.2f}')

    print(f'Average Accuracy Fmask: {avg_accuracy_fmask:.2f}')
    print(f'Average Precision Fmask: {avg_precision_fmask:.2f}')
    print(f'Average Recall Fmask: {avg_recall_fmask:.2f}')
    print(f'Average F1 Score Fmask: {avg_f1_fmask:.2f}')


# Example usage
fmask_folder = "/Volumes/Vault/Fmask_masks/Splited"
norm_folder = "/Volumes/Vault/Splited_set_2_384"

model_path = "/Users/tosha_008/PycharmProjects/cloudFCN_master/models/model_mfcnn_384_30_200_2.keras"
model = load_model(model_path, custom_objects=custom_objects)

display_images(fmask_folder, norm_folder, num_images=15)
