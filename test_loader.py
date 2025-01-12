from tensorflow.keras.models import load_model
from data.Datasets import train_valid_test, LandsatDataset
from data import loader, transformations as trf
from MFCNN.model_mfcnn_def import *
from data.loader import load_paths
from output.defs_for_output import *
import sys, os, pickle, yaml
import numpy as np
 # Project-Specific Imports
# project_path = "/home/ladmin/PycharmProjects/cloudFCN-master"
project_path = "/mnt/agent/system/working_dir"
sys.path.append(project_path)

from data.loader import dataloader_descriptors, generate_descriptor_list_from_yaml, combined_generator
from data import transformations as trf
from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras import backend as K
from SenSei.SenSei_model import Flatten_2D_Op, PermuteDescriptors, Tile_bands_to_descriptor_count, Concatenate_bands_with_descriptors

# Code to load model and dataset remains unchanged
patch_size = 384
bands_1 = [3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11]
bands_2 = [3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12]
batch_size = 10
num_classes = 3
num_channels = len(bands_2)
num_batches_to_show = 1

dataset_path = "/media/ladmin/Vault/Splited_biome_384"  # Biome
dataset_path_2 = '/Volumes/Vault/Splited_set_2_384'  # Set 2 for test
set2_398 = "/media/ladmin/Vault/Splited_set_2_398"  # Set 2 398 for test
dataset_path_2_2 = '/media/ladmin/Vault/Splited_set_2_384'  # Set 2 for test other PC

sentinel_img_dir = "/home/ladmin/PycharmProjects/cloudFCN-master/Sentinel_data/subscenes_splited_384"
sentinel_mask_dir = "/home/ladmin/PycharmProjects/cloudFCN-master/Sentinel_data/masks_splited_384"

test_loader_path = '/home/ladmin/PycharmProjects/cloudFCN-master/for_fine_tuning/cache_train_Biome_96.pkl'
descriptor_path = "/home/ladmin/PycharmProjects/cloudFCN-master/SenSei/landsat_metadata.yaml"


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

sensei_model_paht = "/home/ladmin/PycharmProjects/cloudFCN-master/SenSei/sensei_pretrained_198.h5"

bandsin = Input(shape=(None, None, None, 1))
descriptorsin = Input(shape=(None, 3))
sensei_pretrained = load_model(
    sensei_model_paht,
    custom_objects={  # annoying requirement when using custom layers
        'GroupNormalization': GroupNormalization,
        'Flatten_2D_Op': Flatten_2D_Op,
        'PermuteDescriptors': PermuteDescriptors,
        'Tile_bands_to_descriptor_count': Tile_bands_to_descriptor_count,
        'Concatenate_bands_with_descriptors': Concatenate_bands_with_descriptors,
        'K': K
    })
sensei = sensei_pretrained.get_layer('SEnSeI')
# feature_map = sensei((bandsin, descriptorsin))
# NUM_CHANNELS = sensei.output_shape[-1]


test_set = load_paths(test_loader_path)
descriptor_list = loader.generate_descriptor_list_from_yaml(descriptor_path)

test_ = loader.dataloader_descriptors(
    test_set, batch_size, band_policy=(8),
    transformations=[trf.train_base(patch_size, fixed=True),
                     trf.band_select(),
                     trf.class_merge(3, 4),  # If Biome
                     trf.class_merge(1, 2),  # If Biome
                     trf.change_mask_channels_2_3(),
                     trf.normalize_to_range(),
                     trf.encode_descriptors('log')
                     ],
    shuffle=True,
    descriptor_list=descriptor_list,
    left_mask_channels=2)

images_desc, masks = next(test_())
ims, descriptors = images_desc

print("Images Shape:", ims.shape)  # Expected: (batch_size, patch_size, patch_size, num_channels)
print("Descriptors Shape:", descriptors.shape)
print("Masks Shape:", masks.shape)



descriptors = tf.expand_dims(descriptors, axis=-1)  # (10, 3, 1)
pred = sensei((ims, descriptors))
print(f"Prediction shape: {pred.shape}")

# Assuming `pred` is your prediction tensor with shape (10, 384, 384, 64)

# Loop through each batch

for batch_idx in range(pred.shape[0]):
    print(f"Displaying batch {batch_idx + 1}/{pred.shape[0]}")

    # Extract the batch
    batch = pred[batch_idx].numpy()  # Convert EagerTensor to NumPy array (Shape: (384, 384, 64))

    # Number of channels
    num_channels = batch.shape[-1]

    # Calculate grid size for plotting
    grid_size = int(np.ceil(np.sqrt(num_channels)))

    # Create a figure for all channels
    plt.figure(figsize=(15, 15))

    for channel_idx in range(num_channels):
        plt.subplot(grid_size, grid_size, channel_idx + 1)
        plt.imshow(batch[:, :, channel_idx], cmap='gray')
        plt.title(f'Channel {channel_idx}')
        plt.axis('off')

        # Print min and max values for the current channel
        min_val = batch[:, :, channel_idx].min()
        max_val = batch[:, :, channel_idx].max()
        print(f"Batch {batch_idx}, Channel {channel_idx} - Min value: {min_val}, Max value: {max_val}")

    plt.tight_layout()
    # plt.show()

# for channel in range(ims.shape[-1]):
#     channel_min = ims[..., channel].min()
#     channel_max = ims[..., channel].max()
#     print(f"Channel {channel}: Min = {channel_min}, Max = {channel_max}")
#
#
# last_layer = descriptors[-1]
# min_values = last_layer.min(axis=1)
# max_values = last_layer.max(axis=1)
#
# for i, (min_val, max_val) in enumerate(zip(min_values, max_values)):
#     print(f"Descriptor {i + 1}: Min = {min_val}, Max = {max_val}")
#
# for channel in range(masks.shape[-1]):
#     channel_min = masks[..., channel].min()
#     channel_max = masks[..., channel].max()
#     print(f"Channel masks {channel}: Min = {channel_min}, Max = {channel_max}")


