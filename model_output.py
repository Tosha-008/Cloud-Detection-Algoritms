import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
from cloudFCN.data.Datasets import train_valid_test, LandsatDataset
from cloudFCN.data import loader, transformations as trf


def show_image_mask_and_prediction(image, mask, pred_mask, index):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Show the RGB image
    image_rgb = image[:, :, :3]  # Use only the first 3 channels for RGB
    axes[0].imshow(image_rgb)
    axes[0].set_title(f'Image {index}')
    axes[0].axis('off')

    # Show the original mask
    axes[1].imshow(mask.squeeze(), cmap='gray')
    axes[1].set_title(f'Original Mask {index}')
    axes[1].axis('off')

    # Show the predicted mask
    axes[2].imshow(pred_mask.squeeze(), cmap='gray')
    axes[2].set_title(f'Predicted Mask {index}')
    axes[2].axis('off')

    plt.show()

patch_size = 398
bands = [3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11]
batch_size = 2
num_classes = 3
num_channels = len(bands)
num_batches_to_show = 1

model_path = "/Users/tosha_008/PycharmProjects/cloudFCN-master/models/model_mfcnn_8_250.keras"
dataset_path = "/Volumes/Vault/Splited_data"

# Load the trained model
model = load_model(model_path)

train_path, valid_paths, test_paths = train_valid_test(dataset_path)

test_set = LandsatDataset(test_paths)
test_set.randomly_reduce(0.1)

test_loader = loader.dataloader(
    test_set, batch_size, patch_size,
    transformations=[trf.train_base(patch_size, fixed=True),
                     trf.band_select(bands),
                     trf.class_merge(3, 4),
                     trf.class_merge(1, 2),
                     trf.normalize_to_range()
                     ],
    shuffle=False,
    num_classes=num_classes,
    num_channels=num_channels)

for i, (images, masks) in enumerate(test_loader()):
    if i >= num_batches_to_show:
        break

    print(f"Batch {i + 1}:")
    print(f"Images shape: {images.shape}")
    print(f"Masks shape: {masks.shape}")

    # Make predictions using the loaded model
    predictions = model.predict(images)

    # Loop over images in the batch
    for j in range(images.shape[0]):
        image = images[j]
        mask = masks[j]
        pred_mask = predictions[j]

        show_image_mask_and_prediction(image, mask, pred_mask, i * len(images) + j)


# Загрузите метрики из JSON-файла
# with open('/Users/tosha_008/PycharmProjects/cloudFCN-master/training_history.json', 'r') as f:
#     history = json.load(f)

# # График лосса
# plt.plot(history['loss'], label='train_loss')
# plt.plot(history['val_loss'], label='val_loss')
# plt.title('Loss during training and validation')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# # График точности
# plt.plot(history['categorical_accuracy'], label='train_accuracy')
# plt.plot(history['val_categorical_accuracy'], label='val_accuracy')
# plt.title('Accuracy during training and validation')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
