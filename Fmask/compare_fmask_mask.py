from compare_fmask_mask_defs import *
from tensorflow.keras.models import load_model
from output.defs_for_output import *
import pickle
from data.loader import load_paths, dataloader_descriptors, generate_descriptor_list_from_yaml
from SenSei.SenSei_model import Flatten_2D_Op, PermuteDescriptors, Tile_bands_to_descriptor_count, Concatenate_bands_with_descriptors
from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras import backend as K
from SenSei import spectral_encoders

# Example usage

custom_objects = {
    'MultiscaleLayer': MultiscaleLayer,
    'Up': Up,
    'FMM': FMM,
    'PadByUp': PadByUp,
    'DoubleConv': DoubleConv,
    'ScaleBlock': ScaleBlock,
    'PaddingLayer': PaddingLayer,
    'OutConv': OutConv,
    'GroupNormalization': GroupNormalization,
    'Flatten_2D_Op': Flatten_2D_Op,
    'PermuteDescriptors': PermuteDescriptors,
    'Tile_bands_to_descriptor_count': Tile_bands_to_descriptor_count,
    'Concatenate_bands_with_descriptors': Concatenate_bands_with_descriptors,
    'K': K
}

fmask_folder = "/media/ladmin/Vault/Fmask_masks/Splited"
norm_folder = "/media/ladmin/Vault/Splited_set_2_384"

sentinel_img_dir = "/home/ladmin/PycharmProjects/cloudFCN-master/Sentinel_data/subscenes_splited_384"
sentinel_mask_dir = "/home/ladmin/PycharmProjects/cloudFCN-master/Sentinel_data/masks_splited_384"
test_sentinel_path = "/home/ladmin/PycharmProjects/cloudFCN-master/for_fine_tuning/test_setmfcnn_384_50_200.pkl"

descriptor_path_landsat = "/home/ladmin/PycharmProjects/cloudFCN-master/SenSei/landsat_metadata.yaml"
descriptor_path_sentinel = "/home/ladmin/PycharmProjects/cloudFCN-master/SenSei/sentinel_metadata.yaml"

sentinel_set = None

cloudiness_groups_path_set2 = "/home/ladmin/PycharmProjects/cloudFCN-master/Fmask/cloudiness_groups_set2.pkl"  # Path to the pickle file
cloudiness_groups_path_sentinel = '/home/ladmin/PycharmProjects/cloudFCN-master/Fmask/cloudiness_groups_sentinel.pkl'
cloudiness_groups_path_sentinel_test_small = '/home/ladmin/PycharmProjects/cloudFCN-master/Fmask/cloudiness_groups_sentinel_test_small.pkl'

selected_group = "middle"  # Replace with the desired group name
max_objects = 100

dataset_1 = "Biome"
dataset_2 = "Set_2"
dataset_3 = "Sentinel_2"

model_name_1 = "mfcnn"
model_name_11 = "mfcnn_sentinel"
model_name_111 = 'mfcnn_finetuned'
model_name_1111 = 'mfcnn_finetuned_lowclouds'
model_name_1_2 = 'mfcnn_common'
model_name_2 = "cxn"
model_name_3 = "cloudfcn"
model_name_4 = "sensei_mfcnn"

model_path_1 = '/home/ladmin/PycharmProjects/cloudFCN-master/models/model_mfcnn_384_50_200_2.keras'
model_path_2 = '/home/ladmin/PycharmProjects/cloudFCN-master/models/model_cxn_384_50_200_2.keras'
model_path_3 = '/home/ladmin/PycharmProjects/cloudFCN-master/for_fine_tuning/model_epoch_33_val_loss_0.53.keras'
model_path_4 = '/home/ladmin/PycharmProjects/cloudFCN-master/for_fine_tuning/model_epoch_33_11_finetuned_fine.keras'
model_path_5 = '/home/ladmin/PycharmProjects/cloudFCN-master/for_fine_tuning/model_sentinel_mfcnn_33_11_15_finetuning_lowclouds_2.keras'
model_path_6 = '/home/ladmin/PycharmProjects/cloudFCN-master/mfcnn_common_model/mfcnn_44_commonmodel.keras'
model_path_7 = '/home/ladmin/PycharmProjects/cloudFCN-master/SenSei/sensei_mfcnn_75_300.keras'

main_set = dataset_3
main_model = model_name_4
model = load_model(model_path_7, custom_objects=custom_objects, safe_mode=False)

descriptor_list_landsat = generate_descriptor_list_from_yaml(descriptor_path_landsat)
descriptor_list_sentinel = generate_descriptor_list_from_yaml(descriptor_path_sentinel)

if main_set == "Biome" or main_set == "Set_2":
    descriptors_list = descriptor_list_landsat
else:
    descriptors_list = descriptor_list_sentinel

# if main_set == "Sentinel_2":
    # cloudiness_groups_path = cloudiness_groups_path_sentinel
    # sentinel_set = img_mask_pair(sentinel_img_dir, sentinel_mask_dir)


# output_file_alphas = "alpha_metrics_sensei_mfcnn_landsat_048_049_middle.csv"
# # alpha_values = np.logspace(-4, -1, num=15)
# alpha_values = np.arange(0.48, 0.59, 0.001)
#
# results = crossval_alpha_for_group(
#     pickle_file=cloudiness_groups_path_set2,
#     group_name='middle',
#     norm_folder=norm_folder,
#     fmask_folder=fmask_folder,
#     sentinel_set=None,
#     model=model,
#     model_name=main_model,
#     dataset_name=main_set,
#     alpha_values=alpha_values,
#     output_file=output_file_alphas,
#     max_objects=100,
#     descriptors_list=descriptors_list
# )

metrics_for_all_groups = "metrics_for_all_groups_sensei_mfcnn_sentinel_0485.json"
evaluate_all_groups(
    pickle_file=cloudiness_groups_path_sentinel_test_small,
    output_file=metrics_for_all_groups,
    model=model,
    model_name=main_model,
    dataset_name=main_set,
    fmask_folder=None,
    norm_folder=None,
    max_objects=100,
    descriptors_list=descriptors_list
)

print("Analysis complete. Results saved to", metrics_for_all_groups)


# metrics = evaluate_metrics_for_group(
#     pickle_file=cloudiness_groups_path, 
#     model=model,
#     model_name=main_model,
#     group_name='middle',
#     dataset_name=main_set,
#     fmask_folder=fmask_folder,
#     norm_folder=norm_folder,
#     sentinel_set=sentinel_set,
#     max_objects=5
# )

# display_images_for_group(model=model,
#                          model_name=main_model,
#                          dataset_name=main_set,
#                          group_name='middle',
#                          fmask_folder=None,
#                          norm_folder=None,
#                          max_objects=10,
#                          shuffle=False,
#                          pickle_file=cloudiness_groups_path_sentinel_test_small,  # 'low', 'middle', 'high', 'only clouds', 'no clouds', 'no filter'
#                          display_chanel=None,
#                          predict_uncertainty=False,
#                          T=15,
#                          descriptors_list=descriptors_list)
#
# all_paths_set_landsat = load_paths('/home/ladmin/PycharmProjects/cloudFCN-master/for_fine_tuning/cache_train_Biome_96.pkl')
# output_file_path = "cloudiness_groups_biome.pkl"
# split_images_by_cloudiness(all_paths_set_landsat, output_file_path, dataset_name='Biome',
#                            mask_storage=None)


# output_file_brightness = "brightness_mfcnn_common_500_new.json"
# results = analyze_brightness_contrast(pickle_file=cloudiness_groups_path_set2,
#                             model=model,
#                             dataset_name=main_set,
#                             model_name=main_model,
#                             norm_folder=norm_folder,
#                             max_objects=500,
#                             shuffle=True,
#                             shuffle_seed=42,
#                             selected_groups=None,
#                             output_file=output_file_brightness)
# # Output the results
# for cloud_group, stats in results.items():
#     print(f"Cloud group: {cloud_group}")
#     for stat_name, value in stats.items():
#         print(f"  {stat_name}: {value}")
#     print()