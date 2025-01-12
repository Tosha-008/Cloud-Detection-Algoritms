from tensorflow.keras.models import load_model
from data.Datasets import train_valid_test, LandsatDataset
from data import loader, transformations as trf
from MFCNN.model_mfcnn_def import *
from data.loader import load_paths
from defs_for_output import *


# Code to load model and dataset remains unchanged
patch_size = 384
bands_1 = [3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11]
bands_2 = [3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12]
batch_size = 10
num_classes = 3
num_channels = len(bands_2)
num_batches_to_show = 1

model_path_1 = "/home/ladmin/PycharmProjects/cloudFCN-master/mfcnn_common_model/mfcnn_44_commonmodel.keras"
model_path_2 = '/home/ladmin/PycharmProjects/cloudFCN-master/models/model_mfcnn_384_50_200_2.keras'
metrics_path = "/home/ladmin/PycharmProjects/cloudFCN-master/training_history_sensei_mfcnn_75_300.json"
dataset_path = "/media/ladmin/Vault/Splited_biome_384"  # Biome
dataset_path_2 = '/Volumes/Vault/Splited_set_2_384'  # Set 2 for test
set2_398 = "/media/ladmin/Vault/Splited_set_2_398"  # Set 2 398 for test
dataset_path_2_2 = '/media/ladmin/Vault/Splited_set_2_384'  # Set 2 for test other PC

sentinel_img_dir = "/home/ladmin/PycharmProjects/cloudFCN-master/Sentinel_data/subscenes_splited_384"
sentinel_mask_dir = "/home/ladmin/PycharmProjects/cloudFCN-master/Sentinel_data/masks_splited_384"

test_loader_path = '/home/ladmin/PycharmProjects/cloudFCN-master/for_fine_tuning/cache_train_Biome_96.pkl'

dataset_1 = "Biome"
dataset_2 = "Set_2"
dataset_3 = "Sentinel_2"

model_name_1 = "mfcnn"
model_name_2 = "cloudfcn"
model_name_3 = "cxn"

main_set = dataset_2
main_model = model_name_1

if main_model=='cloudfcn' and main_set=='Set_2':
    test_loader_path = "/home/ladmin/PycharmProjects/cloudFCN-master/output/test_paths_Set_2_35_398.pkl"
    patch_size = 398

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

# model = load_model(model_path_1, custom_objects=custom_objects)


if main_set == "Biome" or main_set == "Set_2":
    test_set = load_paths(test_loader_path)

    if not test_set:
        train_path, valid_paths, test_paths = train_valid_test(set2_398,
                                                               train_ratio=0.7,
                                                               test_ratio=0.1,
                                                               dataset=main_set,
                                                               only_test=True,
                                                               no_test=False)
        test_set = LandsatDataset(test_paths, cache_file=f"test_paths_{main_set}_{len(test_paths)}_398.pkl",
                                  save_cache=True)

    test_ = loader.dataloader(
        test_set, batch_size, patch_size,
        transformations=[trf.train_base(patch_size, fixed=True),
                         # trf.band_select(bands_1),
                         trf.landsat_12_to_13(),
                         trf.class_merge(3, 4),  # If Biome
                         trf.class_merge(1, 2),  # If Biome
                         trf.change_mask_channels_2_3(),
                         # trf.class_merge(2, 3),  # If Set 2
                         # trf.class_merge(0, 1),  # If Set 2
                         trf.normalize_to_range()
                         ],
        shuffle=True,
        num_classes=num_classes,
        num_channels=num_channels,
        left_mask_channels=num_classes)

if main_set == "Sentinel_2":
    sentinel_set = img_mask_pair(sentinel_img_dir, sentinel_mask_dir)

    test_ = loader.dataloader(
        sentinel_set, batch_size, patch_size,
        transformations=[trf.train_base(patch_size, fixed=True),
                         trf.band_select(bands_2),
                         trf.normalize_to_range()
                         ],
        shuffle=True,
        num_classes=num_classes,
        num_channels=num_channels,
        left_mask_channels=num_classes)

# find_alpha(gen=test_,
#            model=model,
#            num_batches_to_show=num_batches_to_show,
#            dataset_name=main_set,
#            model_name=main_model,
#            alpha_values=np.arange(0, 1, 0.05),
#            output_file=f'../model_metrics/alpha_{main_model}_{main_set}_0_1_005.csv')

# count_average_metrics(gen=test_,
#                       model=model,
#                       num_batches_to_show=num_batches_to_show,
#                       dataset_name=main_set,
#                       model_name=main_model)

# plot_batches(gen=test_,
#              model=model,
#              num_batches_to_show=num_batches_to_show,
#              show_masks_pred=True,
#              dataset_name=main_set,
#              model_name=main_model)

# # Plot metrics only if specified
plot_metrics(metrics_path, show_metrics=True)