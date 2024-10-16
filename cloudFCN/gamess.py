import numpy as np
import matplotlib.pyplot as plt

def normalize_to_255(image):
    """
    Normalizes the values of the array to the range from 0 to 255.

    Parameters:
    image (numpy array): The input image (array) to be normalized.

    Returns:
    numpy array: The normalized image with a range of [0, 255].
    """

    min_val = np.min(image)
    max_val = np.max(image)

    normalized_image = 255 * (image - min_val) / (max_val - min_val)

    return normalized_image.astype(np.uint8)

image = np.load('/Volumes/Vault/Splited_data/Snow:Ice/LC80211222013361LGN00/001013/image.npy')

layer_3 = image[..., 3]  # 3-й слой (красный канал)
layer_2 = image[..., 2]  # 2-й слой (зелёный канал)
layer_1 = image[..., 1]  # 1-й слой (синий канал)

rgb_image = np.stack((layer_3, layer_2, layer_1), axis=-1)
print(rgb_image.shape)

plt.imshow(normalize_to_255(rgb_image))
plt.title('Combined Layers 3 (R), 2 (G), 1 (B)')
plt.axis('off')
plt.show()
