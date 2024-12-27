from compare_fmask_mask_defs import *
from tensorflow.keras.models import load_model
from output.defs_for_output import *
import pickle
from data.loader import load_paths

# Example usage

fmask_folder = "/media/ladmin/Vault/Fmask_masks/Splited"
norm_folder = "/media/ladmin/Vault/Splited_set_2_384"

sentinel_img_dir = "/home/ladmin/PycharmProjects/cloudFCN-master/Sentinel_data/subscenes_splited_384"
sentinel_mask_dir = "/home/ladmin/PycharmProjects/cloudFCN-master/Sentinel_data/masks_splited_384"
test_sentinel_path = "/home/ladmin/PycharmProjects/cloudFCN-master/for_fine_tuning/test_setmfcnn_384_50_200.pkl"
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
model_name_2 = "cxn"
model_name_3 = "cloudfcn"

model_path_1 = '/home/ladmin/PycharmProjects/cloudFCN-master/models/model_mfcnn_384_50_200_2.keras'
model_path_2 = '/home/ladmin/PycharmProjects/cloudFCN-master/models/model_cxn_384_50_200_2.keras'
model_path_3 = '/home/ladmin/PycharmProjects/cloudFCN-master/for_fine_tuning/model_epoch_33_val_loss_0.53.keras'
model_path_4 = '/home/ladmin/PycharmProjects/cloudFCN-master/for_fine_tuning/model_epoch_33_11_finetuned_fine.keras'

main_set = dataset_2
main_model = model_name_111
model = load_model(model_path_4, custom_objects=custom_objects)

if main_set == "Sentinel_2":
    # cloudiness_groups_path = cloudiness_groups_path_sentinel
    # sentinel_set = img_mask_pair(sentinel_img_dir, sentinel_mask_dir)
    with open(cloudiness_groups_path_sentinel_test_small, "rb") as f:
        test_sentinel = pickle.load(f)
        # Get the number of elements in each list
        # key_lengths = {key: len(value) for key, value in test_sentinel.items()}
        #
        # # Print the result
        # for key, length in key_lengths.items():
        #     print(f"{key}: {length} elements")


# output_file_alphas = "alpha_metrics_mfcnnsent_landset_finetuned_3311_04_06_low.csv"
# # alpha_values = np.logspace(-4, -1, num=15)
# alpha_values = np.arange(0.4, 0.6, 0.05)
#
# results = crossval_alpha_for_group(
#     pickle_file=cloudiness_groups_path,
#     group_name='low',
#     norm_folder=norm_folder,
#     fmask_folder=fmask_folder,
#     sentinel_set=None,
#     model=model,
#     model_name=main_model,
#     dataset_name=main_set,
#     alpha_values=alpha_values,
#     output_file=output_file_alphas,
#     max_objects=50
# )

# metrics_for_all_groups = "metrics_for_all_groups_mfcnnsent_landset_finetuned_04.json"
# evaluate_all_groups(
#     pickle_file=cloudiness_groups_path,
#     output_file=metrics_for_all_groups,
#     model=model,
#     model_name=main_model,
#     dataset_name=main_set,
#     fmask_folder=fmask_folder,
#     norm_folder=norm_folder,
#     sentinel_set=None,
#     max_objects=100
# )
#
# print("Analysis complete. Results saved to", metrics_for_all_groups)


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
#                          group_name='no clouds',
#                          fmask_folder=fmask_folder,
#                          norm_folder=norm_folder,
#                          sentinel_set=None,
#                          max_objects=10,
#                          shuffle=False,
#                          pickle_file=cloudiness_groups_path) # 'low', 'middle', 'high', 'only clouds', 'no clouds', 'no filter'

# all_paths_set_landsat = load_paths('/home/ladmin/PycharmProjects/cloudFCN-master/for_fine_tuning/cache_train_Biome_96.pkl')
# output_file_path = "cloudiness_groups_biome.pkl"
# split_images_by_cloudiness(all_paths_set_landsat, output_file_path, dataset_name='Biome',
#                            mask_storage=None)
