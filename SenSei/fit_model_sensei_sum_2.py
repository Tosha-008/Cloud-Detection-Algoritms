import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import Input
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras import backend as K
import yaml
import pickle
import json

# Project-Specific Imports
# project_path = "/home/ladmin/PycharmProjects/cloudFCN-master"
project_path = "/mnt/agent/system/working_dir"
sys.path.append(project_path)

from data.loader import dataloader_descriptors, generate_descriptor_list_from_yaml, combined_generator
from data import transformations as trf
from SenSei.callbacks import LearningRateLogger
from SenSei.SenSei_model import Flatten_2D_Op, PermuteDescriptors, Tile_bands_to_descriptor_count, Concatenate_bands_with_descriptors
from MFCNN.model_mfcnn_def import build_model_mfcnn
from SenSei import spectral_encoders


def fit_model_sensei_sum(config):
    """
    Function to train the SEnSeI model using Sentinel-2 and Landsat data.

    Parameters:
        config (dict): Configuration settings from a YAML file.

    Returns:
        model (tf.keras.Model): Trained SEnSeI model.
    """

    # Checking if the dataset exists
    print(os.path.exists(os.path.join(project_path, 'Data_Common_')))

    # Extracting training settings from the config
    model_save_path = config['MODEL_SAVE_PATH']
    model_name = config['MODEL_TYPE']

    batch_size = config['BATCH_SIZE']
    patch_size = config['PATCH_SIZE']
    epochs = config['EPOCHS']
    steps_per_epoch = config['STEPS_PER_EPOCH']

    # Paths to datasets and descriptors
    sentinel_paths = config['SENTINEL_PATHS']
    landsat_paths = config['LANDSAT_PATHS']
    sensei_path = config['SENSEI_PATH']
    descriptor_path_landsat = config['DESCRIPTOR_PATH_LANDSAT']
    descriptor_path_sentinel = config['DESCRIPTOR_PATH_SENTINEL']

    # Range of spectral bands
    BANDS = (3,14)
    classes = config['CLASSES']

    # Defining the final model save path
    model_save_path = os.path.join(model_save_path,
                                   f'sensei_{model_name}_{epochs}_{steps_per_epoch}.keras')

    # Loading descriptor lists
    descriptor_list_landsat = generate_descriptor_list_from_yaml(descriptor_path_landsat)
    descriptor_list_sentinel = generate_descriptor_list_from_yaml(descriptor_path_sentinel)

    # Loading Sentinel-2 and Landsat datasets from pickle files
    with open(sentinel_paths, "rb") as f:
        sentinel_set = pickle.load(f)

    with open(landsat_paths, "rb") as f:
        landsat_set = pickle.load(f)

    # Updating dataset paths
    train_set_sentinel = [
        (
            image_path.replace('/media/ladmin/Vault/Sentinel_2', './Data_Common_/Sentinel_Data'),
            mask_path.replace('/media/ladmin/Vault/Sentinel_2', './Data_Common_/Sentinel_Data')
        )
        for image_path, mask_path in sentinel_set["train_set"]
    ]

    valid_set_sentinel = [
        (
            image_path.replace('/media/ladmin/Vault/Sentinel_2', './Data_Common_/Sentinel_Data'),
            mask_path.replace('/media/ladmin/Vault/Sentinel_2', './Data_Common_/Sentinel_Data')
        )
        for image_path, mask_path in sentinel_set["valid_set"]
    ]
    train_set_landsat = [
        (path.replace('/media/ladmin/Vault/Splited_biome_384', './Data_Common_/Splited_biome_384'))
        for path in landsat_set["train_set"]
    ]
    train_set_landsat = [(f"{path}/image.npy", f"{path}/mask.npy") for path in train_set_landsat]

    valid_set_landsat = [
        (path.replace('/media/ladmin/Vault/Splited_biome_384', './Data_Common_/Splited_biome_384'))
        for path in landsat_set["valid_set"]
    ]
    valid_set_landsat = [(f"{path}/image.npy", f"{path}/mask.npy") for path in valid_set_landsat]

    # Creating data loaders for Sentinel-2 and Landsat with transformations
    train_loader_sentinel = dataloader_descriptors(
        train_set_sentinel, batch_size, band_policy=BANDS,
        transformations=[trf.train_base(patch_size, fixed=False),
                         trf.encode_descriptors('log'),
                         trf.band_select(),
                         trf.sometimes(0.1, trf.salt_and_pepper(0.001, 0.001, pepp_value=-1, salt_value=1)),
                         trf.sometimes(0.3, trf.intensity_scale(0.95, 1.05)),
                         trf.sometimes(0.3, trf.intensity_shift(-0.02, 0.02)),
                         trf.sometimes(0.2, trf.chromatic_scale(0.95, 1.05)),
                         trf.sometimes(0.2, trf.chromatic_shift(-0.02, 0.02)),
                         trf.sometimes(0.3, trf.white_noise(0.1)),
                         trf.sometimes(0.1, trf.quantize(2**6)),
                         trf.normalize_to_range()
                         ],
        shuffle=True,
        descriptor_list=descriptor_list_sentinel,
        left_mask_channels=classes)

    train_loader_landsat = dataloader_descriptors(
        train_set_landsat, batch_size, band_policy=BANDS,
        transformations=[trf.train_base(patch_size, fixed=False),
                         trf.encode_descriptors('log'),
                         trf.band_select(),
                         trf.class_merge(3, 4),
                         trf.class_merge(1, 2),
                         trf.sometimes(0.1, trf.salt_and_pepper(0.001, 0.001, pepp_value=-1, salt_value=1)),
                         trf.sometimes(0.3, trf.intensity_scale(0.95, 1.05)),
                         trf.sometimes(0.3, trf.intensity_shift(-0.02, 0.02)),
                         trf.sometimes(0.2, trf.chromatic_scale(0.95, 1.05)),
                         trf.sometimes(0.2, trf.chromatic_shift(-0.02, 0.02)),
                         trf.sometimes(0.3, trf.white_noise(0.1)),
                         trf.sometimes(0.1, trf.quantize(2 ** 6)),
                         trf.change_mask_channels_2_3(),
                         trf.normalize_to_range()
                         ],
        shuffle=True,
        descriptor_list=descriptor_list_landsat,
        left_mask_channels=classes)


    valid_loader_sentinel = dataloader_descriptors(
        valid_set_sentinel, batch_size, band_policy='all',
        transformations=[trf.train_base(patch_size, fixed=True),
                         trf.encode_descriptors('log'),
                         trf.band_select(),
                         trf.normalize_to_range()
                         ],
        shuffle=False,
        descriptor_list=descriptor_list_sentinel,
        left_mask_channels=classes)


    valid_loader_landsat = dataloader_descriptors(
        valid_set_landsat, batch_size, band_policy='all',
        transformations=[trf.train_base(patch_size, fixed=True),
                         trf.encode_descriptors('log'),
                         trf.band_select(),
                         trf.class_merge(3, 4),
                         trf.class_merge(1, 2),
                         trf.change_mask_channels_2_3(),
                         trf.normalize_to_range()
                         ],
        shuffle=False,
        descriptor_list=descriptor_list_landsat,
        left_mask_channels=classes)

    # Computing validation steps
    total_valid_img = len(valid_set_sentinel) + len(valid_set_landsat)
    landsat_steps = int(0.5 * total_valid_img) // batch_size
    sentinel_steps = int(0.5 * total_valid_img) // batch_size
    summary_steps = landsat_steps + sentinel_steps

    print("Total valid images: {}".format(total_valid_img))
    print('Landsat valid images: {}'.format(len(valid_set_landsat)))
    print('Sentinel valid images: {}'.format(len(valid_set_sentinel)))
    print(f'Summary valid steps: {summary_steps}')

    # Creating data generators
    train_gen_sentinel = train_loader_sentinel()
    valid_gen_sentinel = valid_loader_sentinel()
    train_gen_landsat = train_loader_landsat()
    valid_gen_landsat = valid_loader_landsat()

    mixed_gen_train = combined_generator(train_gen_sentinel, train_gen_landsat, sentinel_weight=0.5, landsat_weight=0.5)
    mixed_gen_valid = combined_generator(valid_gen_sentinel, valid_gen_landsat, sentinel_weight=0.5, landsat_weight=0.5, seed=42)

    # Load pre-trained SEnSeI model if required
    if config['SENSEI']:
        bandsin = Input(shape=(None, None, None, 1))
        descriptorsin = Input(shape=(None, 3))
        sensei_pretrained = load_model(
            sensei_path,
            custom_objects={  # annoying requirement when using custom layers
                'GroupNormalization': GroupNormalization,
                'Flatten_2D_Op': Flatten_2D_Op,
                'PermuteDescriptors': PermuteDescriptors,
                'Tile_bands_to_descriptor_count': Tile_bands_to_descriptor_count,
                'Concatenate_bands_with_descriptors': Concatenate_bands_with_descriptors,
                'K': K
            })
        sensei = sensei_pretrained.get_layer('SEnSeI')
        feature_map = sensei((bandsin, descriptorsin))
        NUM_CHANNELS = sensei.output_shape[-1]

    # Define the main model (MFCNN)
    if model_name == 'mfcnn':
        main_model = build_model_mfcnn(num_channels=NUM_CHANNELS, num_classes=classes, dropout_p=0.5)
    else:
        print('MODEL_TYPE not recognised')
        sys.exit()


    # Define the complete model structure
    if config['SENSEI']:
        outs = main_model(feature_map)
        model = Model(inputs=(bandsin, descriptorsin), outputs=outs)

    logdir = os.path.join(os.path.dirname(model_save_path), 'logs')
    logdir = os.path.join(logdir, config['NAME'])
    modeldir = logdir.replace('logs', 'models')
    os.makedirs(modeldir, exist_ok=True)

    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.25,
        patience=3,
        min_lr=2e-5
    )

    if classes == 2:
        metrics = ['categorical_accuracy', AUC()]

    # optimizer = SGD(learning_rate=0.001, momentum=0.99, nesterov=True)
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=metrics
    )
    # if config['SENSEI']:
        # model.get_layer('SEnSeI').summary(line_length=160)
        # model.get_layer('mfcnn').summary(line_length=160)
    model.summary(line_length=160)

    callback_list = [
        TensorBoard(log_dir=logdir, update_freq='epoch'),
        ModelCheckpoint(
            modeldir + '/{epoch:02d}-{val_loss:.2f}.h5',
            save_weights_only=False,
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
        ModelCheckpoint(
            modeldir + '/latest.h5',
            save_weights_only=False,
            save_best_only=False
        ),
        lr_schedule,
        LearningRateLogger(),
    ]

    # Model training
    history = model.fit(
        mixed_gen_train,
        validation_data=mixed_gen_valid,
        validation_steps=summary_steps,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1,
        callbacks=callback_list,
    )

    try:
        history_path = os.path.join(
            os.path.dirname(model_save_path),
            f'training_history_sensei_{model_name}_{epochs}_{steps_per_epoch}.json')
        os.makedirs(os.path.dirname(history_path), exist_ok=True)

        with open(history_path, 'w') as f:
            json.dump(history.history, f)

        print(f'Training history saved at {history_path}')

    except Exception as e:
        print(f'Training history could not be saved: {e}')

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f'Model saved at {model_save_path}')

    return model

if __name__ == "__main__":
    config_path = sys.argv[1]  # TAKE COMMAND LINE ARGUMENT
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    model = fit_model_sensei_sum(config)