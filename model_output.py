import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os
from cloudFCN.data.Datasets import train_valid_test, LandsatDataset
from cloudFCN.data import loader, transformations as trf
from MFCNN.model_mfcnn_def import *
from cloudFCN.data.loader import load_paths, convert_paths_to_tuples
from cloudFCN.callbacks import calculate_metrics


def show_image_mask_and_prediction(image, mask, pred_mask, index, num_classes, show_masks_pred=False):
    pred_mask_binary = (pred_mask[:, :, -1] > 0.3).astype(float)

    accuracy, precision, recall, f1 = calculate_metrics(mask.squeeze(), pred_mask_binary)

    if show_masks_pred:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        image_rgb = image[:, :, :3]  # Use only the first 3 channels for RGB
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title(f'Image {index}')
        axes[0, 0].axis('off')
        axes[0, 1].imshow(mask.squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title(f'Original Mask {index}')
        axes[0, 1].axis('off')

        for c in range(3):
            pred_layer = pred_mask[:, :, c]
            pred_mask_binary = (pred_layer > 0.5).astype(float)
            axes[1, c].imshow(pred_mask_binary, cmap='gray', vmin=0, vmax=1)
            axes[1, c].set_title(f'Predicted Class {c} Layer')
            axes[1, c].axis('off')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(20, 10))
        image_rgb = image[:, :, :3]  # Use only the first 3 channels for RGB
        axes[0].imshow(image_rgb)
        axes[0].set_title(f'Image {index}')
        axes[0].axis('off')
        axes[1].imshow(mask.squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'Original Mask {index}')
        axes[1].axis('off')

        # Display last channel of pred_mask as a single predicted class layer
        # pred_layer = pred_mask[:, :, -1]
        axes[2].imshow(pred_mask_binary, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title(f'Predicted Class Layer')
        axes[2].axis('off')

    fig.text(0.5, 0.01,
             f"Metrics for Image {index}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1:.4f}",
             ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    plt.show()

    # Print metrics
    print(f"Metrics for Image {index}:")
    print(f" - Accuracy: {accuracy:.4f}")
    print(f" - Precision: {precision:.4f}")
    print(f" - Recall: {recall:.4f}")
    print(f" - F1 Score: {f1:.4f}\n")


def plot_metrics(history, show_metrics=False):
    if show_metrics:
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].plot(history['loss'], label='train_loss')
        axs[0].plot(history['val_loss'], label='val_loss')
        axs[0].set_title('Loss during training and validation')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        axs[1].plot(history['categorical_accuracy'], label='train_accuracy')
        axs[1].plot(history['val_categorical_accuracy'], label='val_accuracy')
        axs[1].set_title('Accuracy during training and validation')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].legend()

        plt.tight_layout()
        plt.show()


# Code to load model and dataset remains unchanged
patch_size = 384
bands = [3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11]
batch_size = 5
num_classes = 1
num_channels = len(bands)
num_batches_to_show = 1

model_path = "/Users/tosha_008/PycharmProjects/cloudFCN-master/models/model_cxn_384_2_5_2.keras"
dataset_path = "/Volumes/Vault/Splited_biome_384"  # Biome
dataset_path_2 = '/Volumes/Vault/Splited_set_2_384'  # Dataset 2 for test
test_loader_path = None
dataset_1 = "Biome"
dataset_2 = "Set_2"

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

model = load_model(model_path, custom_objects=custom_objects)
test_set = load_paths(test_loader_path)

if not test_set:
    train_path, valid_paths, test_paths = train_valid_test(dataset_path_2,
                                                           train_ratio=0.7,
                                                           test_ratio=0.1,
                                                           dataset=dataset_2,
                                                           only_test=True,
                                                           no_test=False)
    test_set = LandsatDataset(test_paths, cache_file=f"test_paths_{dataset_2}_{len(test_paths)}.pkl", save_cache=True)

test_ = loader.dataloader(
    test_set, batch_size, patch_size,
    transformations=[trf.train_base(patch_size, fixed=True),
                     trf.band_select(bands),
                     # trf.class_merge(3, 4),   #  If Biome
                     # trf.class_merge(1, 2),   #  If Biome
                     trf.class_merge(2, 3),  # If Set 2
                     trf.class_merge(0, 1),  # If Set 2
                     trf.normalize_to_range()
                     ],
    shuffle=False,
    num_classes=num_classes,
    num_channels=num_channels,
    remove_mask_chanels=True)  # True if num_classes = 1

for i, (images, masks) in enumerate(test_()):
    if i >= num_batches_to_show:
        break

    print(f"Batch {i + 1}:")
    print(f"Images shape: {images.shape}")
    print(f"Masks shape: {masks.shape}")

    predictions = model.predict(images)

    for j in range(images.shape[0]):
        image = images[j]
        mask = masks[j]
        pred_mask = predictions[j]

        show_image_mask_and_prediction(image, mask, pred_mask, i * len(images) + j, num_classes, show_masks_pred=True)

# Load metrics from JSON file
with open('/Users/tosha_008/PycharmProjects/cloudFCN-master/training_history_cxn_384_2_5.json', 'r') as f:
    history = json.load(f)

# Plot metrics only if specified
plot_metrics(history, show_metrics=False)
