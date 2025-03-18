from compare_fmask_mask_defs import *
from tensorflow.keras.models import load_model
from output.defs_for_output import *
import pickle
from data.loader import load_paths, dataloader_descriptors, generate_descriptor_list_from_yaml
from SenSei.SenSei_model import Flatten_2D_Op, PermuteDescriptors, Tile_bands_to_descriptor_count, Concatenate_bands_with_descriptors
# from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras import backend as K
# from SenSei import spectral_encoders


# Custom objects required for loading the model
custom_objects = {
    'MultiscaleLayer': MultiscaleLayer,
    'Up': Up,
    'FMM': FMM,
    'PadByUp': PadByUp,
    'DoubleConv': DoubleConv,
    'ScaleBlock': ScaleBlock,
    'PaddingLayer': PaddingLayer,
    'OutConv': OutConv,
    # 'GroupNormalization': GroupNormalization,
    'Flatten_2D_Op': Flatten_2D_Op,
    'PermuteDescriptors': PermuteDescriptors,
    'Tile_bands_to_descriptor_count': Tile_bands_to_descriptor_count,
    'Concatenate_bands_with_descriptors': Concatenate_bands_with_descriptors,
    'K': K
}

# Paths to datasets and models
fmask_folder = "/media/ladmin/Vault/Fmask_masks/Splited"
norm_folder = "/media/ladmin/Vault/Splited_set_2_384"

# Sentinel-2 dataset paths
sentinel_img_dir = "/home/ladmin/PycharmProjects/cloudFCN-master/Sentinel_data/subscenes_splited_384"
sentinel_mask_dir = "/home/ladmin/PycharmProjects/cloudFCN-master/Sentinel_data/masks_splited_384"
test_sentinel_path = "/home/ladmin/PycharmProjects/cloudFCN-master/for_fine_tuning/test_setmfcnn_384_50_200.pkl"
sentinel_set = None

# Paths to descriptor files for Landsat and Sentinel
descriptor_path_landsat = "/home/ladmin/PycharmProjects/cloudFCN-master/SenSei/landsat_metadata.yaml"
descriptor_path_sentinel = "/home/ladmin/PycharmProjects/cloudFCN-master/SenSei/sentinel_metadata.yaml"

# Cloudiness groups (pickle files containing classified image groups based on cloud coverage)
cloudiness_groups_path_set2 = "/home/ladmin/PycharmProjects/cloudFCN-master/Fmask/cloudiness_groups_set2.pkl"  # Path to the pickle file
cloudiness_groups_path_sentinel = '/home/ladmin/PycharmProjects/cloudFCN-master/Fmask/cloudiness_groups_sentinel.pkl'
cloudiness_groups_path_sentinel_test_small = '/home/ladmin/PycharmProjects/cloudFCN-master/Fmask/cloudiness_groups_sentinel_test_small.pkl'

# Selected cloudiness group for evaluation
selected_group = "middle"  # Replace with the desired group name

# Number of objects (images) to process
max_objects = 100

# Available datasets
dataset_1 = "Biome"
dataset_2 = "Set_2"
dataset_3 = "Sentinel_2"

# Available model names
model_name_1 = "mfcnn"
model_name_11 = "mfcnn_sentinel"
model_name_111 = 'mfcnn_finetuned'
model_name_1111 = 'mfcnn_finetuned_lowclouds'
model_name_1_2 = 'mfcnn_common'
model_name_2 = "cxn"
model_name_3 = "cloudfcn"
model_name_4 = "sensei_mfcnn"

# Paths to pre-trained models
model_path_1 = '/home/ladmin/PycharmProjects/cloudFCN-master/models/model_mfcnn_384_50_200_2.keras'
model_path_2 = '/home/ladmin/PycharmProjects/cloudFCN-master/models/model_cxn_384_50_200_2.keras'
model_path_3 = '/home/ladmin/PycharmProjects/cloudFCN-master/for_fine_tuning/model_epoch_33_val_loss_0.53.keras'
model_path_4 = '/home/ladmin/PycharmProjects/cloudFCN-master/for_fine_tuning/model_epoch_33_11_finetuned_fine.keras'
model_path_5 = '/home/ladmin/PycharmProjects/cloudFCN-master/for_fine_tuning/model_sentinel_mfcnn_33_11_15_finetuning_lowclouds_2.keras'
model_path_6 = '/home/ladmin/PycharmProjects/cloudFCN-master/mfcnn_common_model/mfcnn_44_commonmodel.keras'
model_path_7 = '/home/ladmin/PycharmProjects/cloudFCN-master/SenSei/sensei_mfcnn_75_300.keras'

# Choose the main dataset and model for evaluation
main_set = dataset_2  # Options: 'Biome', 'Set_2', 'Sentinel_2'
main_model = model_name_1_2  # Options: 'mfcnn', 'mfcnn_common', 'sensei_mfcnn', etc.
model = load_model(model_path_6, custom_objects=custom_objects, safe_mode=True)  # Load the selected model

# descriptor_list_landsat = generate_descriptor_list_from_yaml(descriptor_path_landsat)
# descriptor_list_sentinel = generate_descriptor_list_from_yaml(descriptor_path_sentinel)
#
# if main_set == "Biome" or main_set == "Set_2":
#     descriptors_list = descriptor_list_landsat
# else:
#     descriptors_list = descriptor_list_sentinel

# If the dataset is Sentinel-2, set the appropriate cloudiness group and data paths
if main_set == "Sentinel_2":
    cloudiness_groups_path = cloudiness_groups_path_sentinel
    sentinel_set = img_mask_pair(sentinel_img_dir, sentinel_mask_dir)

# Perform cross-validation to find the best threshold (alpha) for cloud detection
# output_file_alphas = "alpha_metrics_sensei_mfcnn_landsat_048_049_middle.csv"

# Define a range of alpha values for thresholding
# np.logspace(-4, -1, num=15) - alternative logarithmic range
# alpha_values = np.logspace(-4, -1, num=15)
# Here, we use a linear range from 0.48 to 0.59 with a step of 0.001
# alpha_values = np.arange(0.48, 0.59, 0.001)

# Run cross-validation to determine the best alpha value based on F1-score
# results = crossval_alpha_for_group(
#     pickle_file=cloudiness_groups_path_set2,  # Path to cloudiness group classification
#     group_name='middle',  # Select the 'middle' cloudiness group for evaluation
#     norm_folder=norm_folder,
#     fmask_folder=fmask_folder,
#     sentinel_set=None,  # No Sentinel dataset used in this run
#     model=model,  # Load the trained model
#     model_name=main_model,  # Model name (e.g., 'mfcnn_common')
#     dataset_name=main_set,  # Dataset type ('Set_2', 'Biome', etc.)
#     alpha_values=alpha_values,  # Range of alpha values to test
#     output_file=output_file_alphas,  # Output file for results
#     max_objects=100,  # Limit the number of objects to process
#     descriptors_list=descriptors_list  # List of additional descriptors (if needed)
# )


# Evaluate model performance across all cloudiness groups and save results
# metrics_for_all_groups = "metrics_for_all_groups_sensei_mfcnn_landsat_argmax.json"

# Run evaluation for all cloud groups (low, middle, high, only clouds, no clouds)
# evaluate_all_groups(
#     pickle_file=cloudiness_groups_path_set2,  # Cloudiness classification file
#     output_file=metrics_for_all_groups,  # Output file for aggregated metrics
#     model=model,  # Model to evaluate
#     model_name=main_model,  # Model name
#     dataset_name=main_set,  # Dataset type
#     fmask_folder=fmask_folder,  # Path to Fmask images
#     norm_folder=norm_folder,  # Path to normalized images
#     max_objects=100,  # Limit the number of processed images
#     descriptors_list=descriptors_list  # Additional descriptors (if needed)
# )
#
# print("Analysis complete. Results saved to", metrics_for_all_groups)

# Evaluate performance of the model on a specific cloudiness group
# metrics = evaluate_metrics_for_group(
#     pickle_file=cloudiness_groups_path,  # Cloudiness classification file
#     model=model,  # Model for evaluation
#     model_name=main_model,  # Model name
#     group_name='middle',  # Evaluate only the 'middle' cloudiness group
#     dataset_name=main_set,  # Dataset type
#     fmask_folder=fmask_folder,  # Path to Fmask images
#     norm_folder=norm_folder,  # Path to normalized images
#     sentinel_set=sentinel_set,  # Sentinel dataset (if used)
#     max_objects=5  # Limit the number of images processed
# )

# Display images for the selected cloudiness group with predictions and uncertainty estimation
display_images_for_group(
    model=model,
    model_name=main_model,
    dataset_name=main_set,
    group_name='no clouds',  # Choose from: 'low', 'middle', 'high', 'only clouds', 'no clouds', 'no filter'
    fmask_folder=fmask_folder,
    norm_folder=norm_folder,
    max_objects=10,  # Number of images to visualize
    shuffle=True,  # Shuffle the dataset before selecting images
    pickle_file=cloudiness_groups_path_set2,
    display_chanel=1,  # Display the second channel (cloud mask)
    predict_uncertainty=True,  # Enable Monte Carlo dropout for uncertainty estimation
    T=10,  # Number of stochastic forward passes
    descriptors_list=None  # Use None if no additional descriptors are needed
)

# Example: Split images into cloudiness groups and save as a pickle file
# all_paths_set_landsat = load_paths('/home/ladmin/PycharmProjects/cloudFCN-master/for_fine_tuning/cache_train_Biome_96.pkl')
# output_file_path = "cloudiness_groups_biome.pkl"
# split_images_by_cloudiness(all_paths_set_landsat, output_file_path, dataset_name='Biome', mask_storage=None)

# Example: Perform brightness and contrast analysis for cloud masks
# output_file_brightness = "brightness_mfcnn_common_500_new.json"
# results = analyze_brightness_contrast(
#     pickle_file=cloudiness_groups_path_set2,
#     model=model,
#     dataset_name=main_set,
#     model_name=main_model,
#     norm_folder=norm_folder,
#     max_objects=500,
#     shuffle=True,
#     shuffle_seed=42,
#     selected_groups=None,  # Analyze all groups
#     output_file=output_file_brightness
# )

# # Output results
# for cloud_group, stats in results.items():
#     print(f"Cloud group: {cloud_group}")
#     for stat_name, value in stats.items():
#         print(f"  {stat_name}: {value}")
#     print()