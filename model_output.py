import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from cloudFCN.data.Datasets import LandsatDataset
from cloudFCN.data import loader, transformations as trf


def normalize_to_255(image):


    min_val = np.min(image)
    max_val = np.max(image)

    normalized_image = 255 * (image - min_val) / (max_val - min_val)

    return normalized_image.astype(np.uint8)


model_path = "/Users/tosha_008/PycharmProjects/cloudFCN-master/models/epoch0_model_split1.h5"
images = np.load("/Users/tosha_008/PycharmProjects/cloudFCN-master/output_tif/split2/Barren/LC81390292014135LGN00/004013/image.npy")
mask = np.load("/Users/tosha_008/PycharmProjects/cloudFCN-master/output_tif/split2/Barren/LC81390292014135LGN00/004013/mask.npy")

images = images[..., [3, 2, 1, 11]]

# dataset_path = "./output_tif/split2/Barren"

model = load_model(model_path)

# dataset = LandsatDataset(dataset_path)
#
# batch_size = 4
# patch_size = 398
#
# test_loader_gen = loader.dataloader(
#     dataset, batch_size, patch_size,
#     transformations=[trf.train_base(patch_size, fixed=True),
#                      trf.band_select([3, 2, 1, 11]),
#                      trf.class_merge(3, 4),
#                      trf.class_merge(1, 2)],
#     shuffle=True,
#     num_classes=3,
#     num_channels=len([3, 2, 1, 11])
# )()
#
# images, _ = next(test_loader_gen)

images_normalized = normalize_to_255(images)

images_normalized = np.expand_dims(images_normalized, axis=0)

predicted_masks = model.predict(images_normalized)

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(np.squeeze(normalize_to_255(images[..., :3])))
plt.title("Original Image (RGB)")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(np.argmax(mask, axis=-1), cmap='gray')
plt.title("Actual Cloud Mask")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(np.squeeze(np.argmax(predicted_masks, axis=-1)), cmap='gray')
plt.title("Predicted Cloud Mask")
plt.axis('off')

plt.show()

# # График лосса
# plt.plot(model.history['loss'], label='train_loss')
# plt.plot(model.history['val_loss'], label='val_loss')
# plt.title('Loss during training and validation')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# # График точности
# plt.plot(model.history['accuracy'], label='train_accuracy')
# plt.plot(model.history['val_accuracy'], label='val_accuracy')
# plt.title('Accuracy during training and validation')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()