import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import json
from cloudFCN.callbacks import calculate_metrics


def show_image_mask(image, mask):
    fig, axes = plt.subplots(1, 1 + mask.shape[-1], figsize=(15, 5))

    image_rgb = image[:, :, :3]
    axes[0].imshow(image_rgb)
    axes[0].set_title("Image")
    axes[0].axis("off")

    for i in range(mask.shape[-1]):
        axes[i + 1].imshow(mask[:, :, i], cmap="gray", vmin=0, vmax=1)
        axes[i + 1].set_title(f"Mask Channel {i + 1}")
        axes[i + 1].axis("off")

        if i == 1:
            min_val = np.min(mask[:, :, i])
            max_val = np.max(mask[:, :, i])
            print(f"Mask Channel {i + 1}: min = {min_val}, max = {max_val}")

    plt.show()


def img_mask_pair(images_dir, masks_dir):

    image_files = {os.path.basename(file): os.path.join(images_dir, file) for file in os.listdir(images_dir) if file.endswith('.npy') and not file.startswith('._')}
    mask_files = {os.path.basename(file): os.path.join(masks_dir, file) for file in os.listdir(masks_dir) if file.endswith('.npy') and not file.startswith('._')}

    matched_files = [(image_path, mask_files[filename]) for filename, image_path in image_files.items() if filename in mask_files]

    print(f"Found {len(matched_files)} image-mask pairs.")

    return matched_files


def compute_average_metrics(all_accuracies, all_precisions, all_recalls, all_f1s):

    avg_accuracy = np.mean(all_accuracies)
    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)
    avg_f1 = np.mean(all_f1s)
    return avg_accuracy, avg_precision, avg_recall, avg_f1


def show_image_mask_and_prediction(image, mask, pred_mask, index, show_masks_pred=False, dataset_name='Biome', model_name='cloudfcn'):
    if model_name == 'cloudfcn':
        alpha = 0.5
        pred_mask_binary = (pred_mask[:, :, -1] > alpha).astype(float)
    elif model_name == 'mfcnn':
        alpha = 0.6
        pred_mask_binary = (pred_mask[:, :, -1] > alpha).astype(float)
    elif model_name == 'cxn':
        alpha = 0.7
        pred_mask_binary = (pred_mask[:, :, -1] > alpha).astype(float)

    if dataset_name == 'Biome':
        accuracy, precision, recall, f1 = calculate_metrics(mask[:, :, -1].squeeze(), pred_mask_binary)
    elif dataset_name == 'Set_2':
        accuracy, precision, recall, f1 = calculate_metrics(mask[:, :, -1].squeeze(), pred_mask_binary)
    elif dataset_name == 'Sentinel_2':
        accuracy, precision, recall, f1 = calculate_metrics(mask[:, :, -2].squeeze(), pred_mask_binary)
        mask = mask[:, :, -2]

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
            # pred_mask_binary = (pred_layer < alpha).astype(float)
            axes[1, c].imshow(pred_layer, cmap='gray', vmin=0, vmax=1)
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


def plot_metrics(path, show_metrics=False):
    if show_metrics:
        with open(path, 'r') as f:
            history = json.load(f)
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


def count_average_metrics(gen, model, num_batches_to_show, dataset_name='Biome', model_name='cloudfcn'):
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1s = []

    for i, (images, masks) in enumerate(gen()):
        if i >= num_batches_to_show:
            break

        print(f"Batch {i + 1}:")

        predictions = model.predict(images)

        for j in range(images.shape[0]):
            mask = masks[j]
            pred_mask = predictions[j]

            if model_name == 'cloudfcn':
                pred_mask_binary = (pred_mask[:, :, -2] < 0.95).astype(float)
            elif model_name == 'mfcnn':
                alpha = 0.6
                pred_mask_binary = (pred_mask[:, :, -1] > alpha).astype(float)
            elif model_name == 'cxn':
                alpha = 0.7
                pred_mask_binary = (pred_mask[:, :, -1] > alpha).astype(float)

            if dataset_name == 'Biome':
                accuracy, precision, recall, f1 = calculate_metrics(mask[:, :, -1].squeeze(), pred_mask_binary)
            elif dataset_name == 'Set_2':
                accuracy, precision, recall, f1 = calculate_metrics(mask[:, :, -1].squeeze(), pred_mask_binary)
            elif dataset_name == 'Sentinel_2':
                accuracy, precision, recall, f1 = calculate_metrics(mask[:, :, -2].squeeze(), pred_mask_binary)

            all_accuracies.append(accuracy)
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)

    avg_accuracy, avg_precision, avg_recall, avg_f1 = compute_average_metrics(
        all_accuracies, all_precisions, all_recalls, all_f1s
    )

    print("Average Metrics for the batch:")
    print(f" - Accuracy: {avg_accuracy:.4f}")
    print(f" - Precision: {avg_precision:.4f}")
    print(f" - Recall: {avg_recall:.4f}")
    print(f" - F1 Score: {avg_f1:.4f}")

def plot_batches(gen, model, num_batches_to_show, show_masks_pred=False, dataset_name='Biome', model_name='cloudfcn'):
    for i, (images, masks) in enumerate(gen()):
        if i >= num_batches_to_show:
            break

        print(f"Batch {i + 1}:")

        predictions = model.predict(images)

        for j in range(images.shape[0]):
            image = images[j]
            mask = masks[j]
            pred_mask = predictions[j]

            show_image_mask_and_prediction(image,
                                           mask,
                                           pred_mask,
                                           i * len(images) + j,
                                           show_masks_pred=show_masks_pred,
                                           dataset_name=dataset_name,
                                           model_name=model_name)



def find_alpha(gen, model, num_batches_to_show, dataset_name='Biome', model_name='cloudfcn',
               alpha_values=None, output_file='alpha_metrics.csv'):
    if alpha_values is None:
        alpha_values = [0.5, 0.7, 0.95]
    best_alpha = None
    best_metrics = {
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0
    }
    best_avg_f1 = 0

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Alpha', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

    for alpha in alpha_values:
        print(f"\nEvaluating for alpha = {alpha}...")

        all_accuracies = []
        all_precisions = []
        all_recalls = []
        all_f1s = []

        for i, (images, masks) in enumerate(gen()):
            if i >= num_batches_to_show:
                break

            predictions = model.predict(images)

            for j in range(images.shape[0]):
                mask = masks[j]
                pred_mask = predictions[j]

                # Apply threshold based on alpha and model_name
                if model_name == "cloudfcn":
                    pred_mask_binary = (pred_mask[:, :, -1] < alpha).astype(float)
                elif model_name == "mfcnn":
                    pred_mask_binary = (pred_mask[:, :, -1] > alpha).astype(float)
                elif model_name == "cxn":
                    pred_mask_binary = (pred_mask[:, :, -1] > alpha).astype(float)
                # Select the right layer based on dataset_name
                if dataset_name == "Biome" :
                    accuracy, precision, recall, f1 = calculate_metrics(mask[:, :, -1].squeeze(), pred_mask_binary)
                elif dataset_name == "Set_2":
                    accuracy, precision, recall, f1 = calculate_metrics(mask[:, :, -1].squeeze(), pred_mask_binary)
                elif dataset_name == "Sentinel_2":
                    accuracy, precision, recall, f1 = calculate_metrics(mask[:, :, -2].squeeze(), pred_mask_binary)
                all_accuracies.append(accuracy)
                all_precisions.append(precision)
                all_recalls.append(recall)
                all_f1s.append(f1)

        # Compute the average metrics for the current alpha
        avg_accuracy, avg_precision, avg_recall, avg_f1 = compute_average_metrics(
            all_accuracies, all_precisions, all_recalls, all_f1s
        )

        print(f"Metrics for alpha = {alpha}:")
        print(f" - Accuracy: {avg_accuracy:.4f}")
        print(f" - Precision: {avg_precision:.4f}")
        print(f" - Recall: {avg_recall:.4f}")
        print(f" - F1 Score: {avg_f1:.4f}")

        with open(output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([alpha, avg_accuracy, avg_precision, avg_recall, avg_f1])

        # Update the best alpha if this alpha has a higher average F1 score
        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            best_alpha = alpha
            best_metrics['accuracy'] = avg_accuracy
            best_metrics['precision'] = avg_precision
            best_metrics['recall'] = avg_recall
            best_metrics['f1'] = avg_f1

    # Output the best alpha and its metrics
    print("\nBest Alpha and Corresponding Metrics:")
    print(f"Best Alpha: {best_alpha}")
    print(f" - Accuracy: {best_metrics['accuracy']:.4f}")
    print(f" - Precision: {best_metrics['precision']:.4f}")
    print(f" - Recall: {best_metrics['recall']:.4f}")
    print(f" - F1 Score: {best_metrics['f1']:.4f}")
