# Standard Library Imports
import os
import sys
import json
import pickle
import random

# Third-Party Library Imports
from tensorflow.keras.optimizers import Adadelta, Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

# Project-Specific Imports
project_path = "/home/ladmin/PycharmProjects/cloudFCN-master"
# project_path = "/mnt/agent/system/working_dir"
sys.path.append(project_path)

from data import loader, transformations as trf
from MFCNN import model_mfcnn_def



def fine_tuning(config):
    """
    Return trained keras model. Main training function for cloud detection. Parameters
    contained in config file.
    """
    io_opts = config['io_options']
    model_save_path = io_opts['model_save_path']
    model_name = io_opts['model_name']

    fit_opts = config['fit_options']
    batch_size = fit_opts['batch_size']
    patch_size = fit_opts['patch_size']
    epochs = fit_opts['epochs']
    steps_per_epoch = fit_opts['steps_per_epoch']
    num_classes = config['model_options']['num_classes']
    bands = config['model_options']['bands']

    # sentinel_img_dir = io_opts['sentinel_img_path']
    # sentinel_mask_dir = io_opts['sentinel_mask_path']
    sentinel_paths = io_opts['sentinel_paths']
    data_path_landsat = io_opts['data_path_landsat']
    model_load_path = io_opts['model_load_path']
    fine_tune = fit_opts['fine_tune']

    model_save_path = os.path.join(model_save_path,
                                   f'model_{model_name}_{epochs}_{steps_per_epoch}_commonmodel_lowclouds.keras')

    if bands is not None:
        num_channels = len(bands)
    else:
        num_channels = 13

    with open(sentinel_paths, "rb") as f:
        sentinel_set = pickle.load(f)

    with open(data_path_landsat, "rb") as f:
        landsat_set = pickle.load(f)

    train_set_sentinel = [
        (
            image_path.replace('/media/ladmin/Vault/Sentinel_2', './Sentinel_data'),
            mask_path.replace('/media/ladmin/Vault/Sentinel_2', './Sentinel_data')
        )
        for image_path, mask_path in sentinel_set["train_set"]
    ]

    valid_set_sentinel = [
        (
            image_path.replace('/media/ladmin/Vault/Sentinel_2', './Sentinel_data'),
            mask_path.replace('/media/ladmin/Vault/Sentinel_2', './Sentinel_data')
        )
        for image_path, mask_path in sentinel_set["valid_set"]
    ]
    train_set_landsat = [
        (path.replace('/media/ladmin/Vault/Splited_biome_384', './landsat_data'))
        for path in landsat_set["train_set"]
    ]
    train_set_landsat = [(f"{path}/image.npy", f"{path}/mask.npy") for path in train_set_landsat]

    valid_set_landsat = [
        (path.replace('/media/ladmin/Vault/Splited_biome_384', './landsat_data'))
        for path in landsat_set["valid_set"]
    ]
    valid_set_landsat = [(f"{path}/image.npy", f"{path}/mask.npy") for path in valid_set_landsat]

    train_loader_sentinel = loader.dataloader(
        train_set_sentinel, batch_size, patch_size,
        transformations=[trf.train_base(patch_size, fixed=False),
                         trf.band_select(bands),
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
        num_classes=num_classes,
        num_channels=num_channels,
        left_mask_channels=num_classes)

    train_loader_landsat = loader.dataloader(
        train_set_landsat, batch_size, patch_size,
        transformations=[trf.train_base(patch_size, fixed=False),
                         trf.landsat_12_to_13(),
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
        num_classes=num_classes,
        num_channels=num_channels,
        left_mask_channels=num_classes)


    valid_loader_sentinel = loader.dataloader(
        valid_set_sentinel, batch_size, patch_size,
        transformations=[trf.train_base(patch_size, fixed=True),
                         trf.band_select(bands),
                         trf.normalize_to_range()
                         ],
        shuffle=False,
        num_classes=num_classes,
        num_channels=num_channels,
        left_mask_channels=num_classes)


    valid_loader_landsat = loader.dataloader(
        valid_set_landsat, batch_size, patch_size,
        transformations=[trf.train_base(patch_size, fixed=True),
                         trf.landsat_12_to_13(),
                         trf.class_merge(3, 4),
                         trf.class_merge(1, 2),
                         trf.change_mask_channels_2_3(),
                         trf.normalize_to_range()
                         ],
        shuffle=False,
        num_classes=num_classes,
        num_channels=num_channels,
        left_mask_channels=num_classes)

    total_valid_img = len(valid_set_sentinel) + len(valid_set_landsat)
    landsat_steps = int(0.5 * total_valid_img) // batch_size
    sentinel_steps = int(0.5 * total_valid_img) // batch_size
    summary_steps = landsat_steps + sentinel_steps
    print("Total valid images: {}".format(total_valid_img))
    print('Landsat valid images: {}'.format(len(valid_set_landsat)))
    print('Sentinel valid images: {}'.format(len(valid_set_sentinel)))
    print(f'Summary steps: {summary_steps}')

    train_gen_sentinel = train_loader_sentinel()
    valid_gen_sentinel = valid_loader_sentinel()
    train_gen_landsat = train_loader_landsat()
    valid_gen_landsat = valid_loader_landsat()

    mixed_gen_train = loader.combined_generator(train_gen_sentinel, train_gen_landsat, sentinel_weight=0.5, landsat_weight=0.5)
    mixed_gen_valid = loader.combined_generator(valid_gen_sentinel, valid_gen_landsat, sentinel_weight=0.5, landsat_weight=0.5, seed=42)

    csv_logger_save_root = os.path.join(
        os.path.dirname(model_save_path), f'training_log_commonmodel_lowclouds_{model_name}_{epochs}.csv'
    )
    model_checkpoint_save_root = os.path.join(
        os.path.dirname(model_save_path), 'model_epoch_{epoch:02d}_val_loss_{val_loss:.2f}_commonmodel_lowclouds.keras'
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Callbacks
    csv_logger = CSVLogger(csv_logger_save_root, append=True)
    model_checkpoint = ModelCheckpoint(
        model_checkpoint_save_root,
        monitor='val_loss',  # Assuming you're monitoring validation loss
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    # early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    # Combine callbacks
    callbacks = [model_checkpoint, csv_logger]

    if model_load_path:
        model = load_model(model_load_path)
    elif model_name == "mfcnn" and not fine_tune:
        model = model_mfcnn_def.build_model_mfcnn(
            num_channels=num_channels, num_classes=num_classes, dropout_p=0.5)
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy']
        )
        model.summary()
    else:
        raise ValueError('Choose correct model`s name')

    if fine_tune:
        for layer in model.layers:
                layer.trainable = True
        for layer in model.layers:
            if layer.name in ['fmm', 'multiscale_layer', 'pad_by_up', 'pad_by_up_1', 'pad_by_up_2', 'dropout', 'activation', 'input_layer']:
                layer.trainable = False
            else:
                layer.trainable = True
        for i, layer in enumerate(model.layers):
            print(f"Layer {i}: {layer.name}, Trainable: {layer.trainable}")

        optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy']
        )
        model.summary()

    history = model.fit(
        mixed_gen_train,
        validation_data=mixed_gen_valid,
        validation_steps=summary_steps,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks)

    try:
        history_path = os.path.join(
            os.path.dirname(model_save_path),
            f'training_history_{model_name}_{epochs}_{steps_per_epoch}_commonmodel_lowclouds.json'
        )
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
        config = json.load(f)
    model = fine_tuning(config)




