from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adagrad, SGD, Adadelta, Adam

import json
import sys
import os
import shutil

project_path = "/content/cloudFCN-master"
# project_path = "/Users/tosha_008/PycharmProjects/cloudFCN-master"
sys.path.append(project_path)

# OUR STUFF
from cloudFCN.data import loader, transformations as trf
from cloudFCN.data.Datasets import LandsatDataset, train_valid_test, randomly_reduce_list
from cloudFCN import models, callbacks
from cloudFCN.experiments import custom_callbacks
from MFCNN import model_mfcnn_def
from cxn import cxn_model
from cloudFCN.data.loader import load_paths


def get_model(config):
    """
    Create and return a compiled Keras model based on the configuration.
    """
    io_opts = config['io_options']
    model_load_path = io_opts['model_load_path']
    model_name = io_opts['model_name']

    # Model options
    num_classes = config['model_options']['num_classes']
    bands = config['model_options']['bands']
    num_channels = len(bands) if bands is not None else 12

    if model_name == "cloudfcn":
        model = models.build_model5(
            batch_norm=True, num_channels=num_channels, num_classes=num_classes)
        optimizer = Adadelta()
        model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer=optimizer)
    elif model_name == "mfcnn":
        model = model_mfcnn_def.build_model_mfcnn(
            num_channels=num_channels, num_classes=num_classes)
        optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)
        model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer=optimizer)
    elif model_name == "cxn":
        model = cxn_model.model_arch(input_rows=config['fit_options']['patch_size'],
                                     input_cols=config['fit_options']['patch_size'],
                                     num_of_channels=num_channels,
                                     num_of_classes=num_classes)
        optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)
        model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer=optimizer)
    else:
        raise ValueError('Choose correct model name')

    model.summary()
    return model

def fit_model(config):
    """
    Return trained keras model. Main training function for cloud detection. Parameters
    contained in config file.
    """
    io_opts = config['io_options']
    model_load_path = io_opts['model_load_path']
    model_save_path = io_opts['model_save_path']
    model_checkpoint_dir = io_opts['model_checkpoint_dir']
    model_name = io_opts['model_name']

    fit_opts = config['fit_options']
    batch_size = fit_opts['batch_size']
    patch_size = fit_opts['patch_size']
    epochs = fit_opts['epochs']
    steps_per_epoch = fit_opts['steps_per_epoch']
    num_classes = config['model_options']['num_classes']
    bands = config['model_options']['bands']
    summary_valid_percent = io_opts['summary_valid_percent']
    dataset_name = io_opts['dataset_name']

    model_save_path = os.path.join(model_save_path,
                                   f'model_{model_name}_{patch_size}_{epochs}_{steps_per_epoch}_2.keras')
    data_path = io_opts['data_path']
    train_loader_path = io_opts["train_loader_path"]
    valid_loader_path = io_opts["valid_loader_path"]
    test_loader_path = io_opts["test_loader_path"]

    # os.makedirs('/content/train_part_set', exist_ok=True)
    drive_prefix = '/content/drive/MyDrive/Splited_data'
    local_train_prefix = '/content/local_train_data'
    local_valid_prefix = '/content/local_valid_data'

    if model_checkpoint_dir is not None:
        chkpnt_path = os.path.join(
            model_checkpoint_dir, 'weights_2.{epoch:02d}-{val_loss:.2f}.keras')

    if bands is not None:
        num_channels = len(bands)
    else:
        num_channels = 12

    # Loading of saved datasets
    train_set = load_paths(train_loader_path)
    summary_valid_set, valid_paths = load_paths(valid_loader_path, valid=True)
    if valid_paths:
        valid_paths = list(dict.fromkeys(os.path.dirname(path) for path in valid_paths))
        lokal_valid_paths = [i.replace('/Volumes/Vault/Splited_biome_384', local_valid_prefix) for i in valid_paths]
    test_set = load_paths(test_loader_path)

    # Loading and downloading valid dataset to Colab

    updated_sum_valid_paths = []
    valid_dirs_to_make = []

    for image_path, mask_path in summary_valid_set:
        new_image_path = image_path.replace('/Volumes/Vault/Splited_biome_384', drive_prefix)
        new_mask_path = mask_path.replace('/Volumes/Vault/Splited_biome_384', drive_prefix)
        updated_sum_valid_paths.extend((new_image_path, new_mask_path))
        valid_dirs_to_make.extend(
            os.path.dirname(image_path).replace('/Volumes/Vault/Splited_biome_384', local_valid_prefix))

    print("Loading validation data:")
    for indx, files in enumerate(updated_sum_valid_paths):
        local_sum_valid_paths = []
        drive_image_path, drive_mask_path = files
        local_image_path = os.path.join(valid_dirs_to_make[indx], os.path.basename(drive_image_path))
        local_mask_path = os.path.join(valid_dirs_to_make[indx], os.path.basename(drive_mask_path))
        os.makedirs(valid_dirs_to_make[indx], exist_ok=True)

        # !cp
        # '{valid_dirs_to_make}' '{local_image_path}'
        # !cp
        # '{valid_dirs_to_make}' '{local_mask_path}'

        local_sum_valid_paths.extend((local_image_path, local_mask_path))
    print(f"Loading validation data COMPLETED")
    print(local_sum_valid_paths)
    foga_valid_sets = [LandsatDataset(valid_path, save_cache=False) for valid_path in lokal_valid_paths]
    foga_valid_loaders = [
        loader.dataloader(
            valid_set, batch_size, patch_size,
            transformations=[trf.train_base(patch_size, fixed=True),
                             trf.band_select(bands),
                             trf.class_merge(3, 4),
                             trf.class_merge(1, 2),
                             trf.normalize_to_range()
                             ],
            shuffle=False,
            num_classes=num_classes,
            num_channels=num_channels,
            remove_mask_chanels=False) for valid_set in foga_valid_sets]

    summary_valid_set = randomly_reduce_list(local_sum_valid_paths, summary_valid_percent)
    summary_batch_size = 12
    summary_steps = len(summary_valid_set) // summary_batch_size
    summary_valid_loader = loader.dataloader(
        summary_valid_set, summary_batch_size, patch_size,
        transformations=[trf.train_base(patch_size, fixed=True),
                         trf.band_select(bands),
                         trf.class_merge(3, 4),
                         trf.class_merge(1, 2),
                         trf.normalize_to_range()
                         ],
        shuffle=False,
        num_classes=num_classes,
        num_channels=num_channels,
        remove_mask_chanels=False)

    # Loading and downloading TRAIN dataset to Colab

    updated_train_paths = []
    dirs_to_make = []

    for image_path, mask_path in train_set:
        new_image_path = image_path.replace('/Volumes/Vault/Splited_biome_384', drive_prefix)
        new_mask_path = mask_path.replace('/Volumes/Vault/Splited_biome_384', drive_prefix)
        updated_train_paths.append((new_image_path, new_mask_path))
        dirs_to_make.append(os.path.dirname(image_path).replace('/Volumes/Vault/Splited_biome_384', local_train_prefix))

    num_parts = 7
    part_size = len(updated_train_paths) // num_parts
    train_data_parts = [updated_train_paths[i * part_size:(i + 1) * part_size] for i in range(num_parts)]

    if model_load_path and os.path.exists(model_load_path):
        print("Loading model from", model_load_path)
        model = load_model(model_load_path)
    else:
        print("Creating new model")
        model = get_model(config)

    summary_valid_gen = summary_valid_loader()
    foga_valid_gens = [foga_valid_loader()
                       for foga_valid_loader in foga_valid_loaders]
    callback_list = [custom_callbacks.foga_table5_Callback_no_thin(
        foga_valid_sets, foga_valid_gens, frequency=1)]
    if model_checkpoint_dir is not None:
        callback_list.append(ModelCheckpoint(chkpnt_path, monitor='val_loss', verbose=0,
                                             save_best_only=True, save_weights_only=False,
                                             save_freq='epoch'))


    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        if os.path.exists('/content/local_train_data'):
            shutil.rmtree('/content/local_train_data')
            print(f"Временная директория '{'/content/local_train_data'}' удалена перед загрузкой новой партии.")

        local_paths = []
        for indx, files in enumerate(train_data_parts[epoch]):
            print(f"Loading data for part {epoch + 1}:")
            drive_image_path, drive_mask_path = files
            local_image_path = os.path.join(dirs_to_make[indx], os.path.basename(drive_image_path))
            local_mask_path = os.path.join(dirs_to_make[indx], os.path.basename(drive_mask_path))
            os.makedirs(dirs_to_make[indx], exist_ok=True)

            # !cp
            # '{drive_image_path}' '{local_image_path}'
            # !cp
            # '{drive_mask_path}' '{local_mask_path}'

            local_paths.extend((local_image_path, local_mask_path))
            print(f"Loading data for part {epoch + 1} COMPLETED")

        train_loader = loader.dataloader(
            local_paths, batch_size, patch_size,
            transformations=[trf.train_base(patch_size),
                             trf.band_select(bands),
                             trf.class_merge(3, 4),
                             trf.class_merge(1, 2),
                             trf.normalize_to_range()
                             ],
            shuffle=True,
            num_classes=num_classes,
            num_channels=num_channels,
            remove_mask_chanels=False)

        train_gen = train_loader()

        model.fit(train_gen,
                  epochs=1,
                  validation_data=summary_valid_gen,
                  validation_steps=summary_steps,
                  steps_per_epoch=steps_per_epoch,
                  verbose=1,
                  callbacks=callback_list
        )

    with open(f'training_history_{model_name}_{patch_size}_{epochs}_{steps_per_epoch}.json', 'w') as f:
        json.dump(model.history, f)

    model.save(model_save_path)
    return model


if __name__ == "__main__":
    config_path = sys.argv[1]  # TAKE COMMAND LINE ARGUMENT
    with open(config_path, 'r') as f:
        config = json.load(f)
    model = fit_model(config)
