from compare_fmask_mask_defs import *
from tensorflow.keras.models import load_model
from output.defs_for_output import *

# Example usage

fmask_folder = "/media/ladmin/Vault/Fmask_masks/Splited"
norm_folder = "/media/ladmin/Vault/Splited_set_2_384"

sentinel_img_dir = "/media/ladmin/Vault/Sentinel_2/subscenes_splited_384"
sentinel_mask_dir = "/media/ladmin/Vault/Sentinel_2/masks_splited_384"
sentinel_set = None

cloudiness_groups_path = "/home/ladmin/PycharmProjects/cloudFCN-master/Fmask/cloudiness_groups.pkl"  # Path to the pickle file
cloudiness_groups_path_sentinel = '/home/ladmin/PycharmProjects/cloudFCN-master/Fmask/cloudiness_groups_sentinel.pkl'

selected_group = "middle"  # Replace with the desired group name
max_objects = 100

dataset_1 = "Biome"
dataset_2 = "Set_2"
dataset_3 = "Sentinel_2"

model_name_1 = "mfcnn"
model_name_2 = "cxn"
model_name_3 = "cloudfcn"

model_path_1 = '/home/ladmin/PycharmProjects/cloudFCN-master/models/model_mfcnn_384_50_200_2.keras'
model_path_2 = '/home/ladmin/PycharmProjects/cloudFCN-master/models/model_cxn_384_50_200_2.keras'

main_set = dataset_3
main_model = model_name_1
model = load_model(model_path_1, custom_objects=custom_objects)

if main_set == "Sentinel_2":
    cloudiness_groups_path = cloudiness_groups_path_sentinel
    sentinel_set = img_mask_pair(sentinel_img_dir, sentinel_mask_dir)




# output_file_alphas = "alpha_metrics_cxn_017_024_middle.csv"
# alpha_values = np.logspace(-12, -21, num=10, base=10)
# alpha_values = np.arange(0.17, 0.24, 0.01)

# results = crossval_alpha_for_group(
#     pickle_file=cloudiness_groups_path,
#     group_name='middle',
#     fmask_folder=fmask_folder,
#     norm_folder=norm_folder,
#     model=model,
#     model_name=model_name,
#     dataset_name=dataset_name,
#     alpha_values=alpha_values,
#     output_file=output_file_alphas,
#     max_objects=max_objects
# )

# metrics_for_all_groups = "metrics_for_all_groups_cxn_017.json"
# evaluate_all_groups(
#     pickle_file=cloudiness_groups_path,
#     fmask_folder=fmask_folder,
#     norm_folder=norm_folder,
#     model=model,
#     model_name=main_model,
#     dataset_name=main_set,
#     output_file=metrics_for_all_groups,
#     max_objects=max_objects
# )
#
# print("Analysis complete. Results saved to", metrics_for_all_groups)


# metrics = evaluate_metrics_for_group(
#     pickle_file=cloudiness_groups_path,
#     fmask_folder=fmask_folder,
#     norm_folder=norm_folder,
#     model=model,
#     model_name=model_name,
#     group_name='low',
#     dataset_name=dataset_name,
#     max_objects=1
# )


display_images_for_group(model=model,
                         model_name=main_model,
                         dataset_name=main_set,
                         group_name='middle',
                         fmask_folder=fmask_folder,
                         norm_folder=norm_folder,
                         sentinel_set=sentinel_set,
                         num_images=20,
                         pickle_file=cloudiness_groups_path) # 'low', 'middle', 'high', 'only clouds', 'no clouds', 'no filter'

# output_file_path = "cloudiness_groups_sentinel.pkl"
# split_images_by_cloudiness(sentinel_set, output_file_path, dataset_name=main_set,
#                            mask_storage=None)
