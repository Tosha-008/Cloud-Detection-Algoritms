import numpy as np
import matplotlib.pyplot as plt

def normalize_to_255(image):
    """
    Нормализует значения массива в диапазон от 0 до 255.

    Parameters:
    image (numpy array): Исходное изображение (массив), которое нужно нормализовать.

    Returns:
    numpy array: Нормализованное изображение с диапазоном [0, 255].
    """

    min_val = np.min(image)
    max_val = np.max(image)

    normalized_image = 255 * (image - min_val) / (max_val - min_val)

    return normalized_image.astype(np.uint8)

image = np.load('/Users/tosha_008/PycharmProjects/cloudFCN-master/output_tif/split2/Barren/LC81390292014135LGN00/004013/image.npy')

layer_3 = image[..., 3]  # 3-й слой (красный канал)
layer_2 = image[..., 2]  # 2-й слой (зелёный канал)
layer_1 = image[..., 1]  # 1-й слой (синий канал)

rgb_image = np.stack((layer_3, layer_2, layer_1), axis=-1)

plt.imshow(normalize_to_255(rgb_image))
plt.title('Combined Layers 3 (R), 2 (G), 1 (B)')
plt.axis('off')
plt.show()

# num_layers = image.shape[-1]

# for i in range(num_layers):
#     plt.figure(figsize=(5, 5))
#     plt.imshow(image[..., i], cmap='gray')  # Отображаем каждый слой в оттенках серого
#     plt.title(f'Layer {i + 1}')
#     plt.axis('off')
#     plt.show()
