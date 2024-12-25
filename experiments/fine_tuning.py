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
project_path = "/mnt/agent/system/working_dir"
sys.path.append(project_path)
from data import loader, transformations as trf
from data.Datasets import train_valid_test_sentinel
from output.defs_for_output import img_mask_pair


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

    sentinel_img_dir = io_opts['sentinel_img_path']
    sentinel_mask_dir = io_opts['sentinel_mask_path']
    test_sentinel_path = io_opts['test_sentinel_path']
    data_path_landsat = io_opts['data_path_landsat']
    model_load_path = io_opts['model_load_path']

    model_save_path = os.path.join(model_save_path,
                                   f'model_sentinel_{model_name}_{patch_size}_{epochs}_{steps_per_epoch}_finetuning.keras')

    if bands is not None:
        num_channels = len(bands)
    else:
        num_channels = 13

    # SENTINEL GEN
    sentinel_set = img_mask_pair(sentinel_img_dir, sentinel_mask_dir)

    with open(test_sentinel_path, "rb") as f:
        test_sentinel = pickle.load(f)

    test_sentinel_full_paths = []
    for image_path, mask_path in test_sentinel:
        # Remove './' and prepend the root directory
        image_clean_path = image_path.lstrip('./')
        mask_clean_path = mask_path.lstrip('./')
        adjusted_image_path = os.path.join(project_path, image_clean_path)
        adjusted_mask_path = os.path.join(project_path, mask_clean_path)
        test_sentinel_full_paths.append((adjusted_image_path, adjusted_mask_path))


    pickle_paths_set = set(sentinel_set)

    # Filter out tuples that are in the pickle file
    sentinel_set_no_repitition = [path for path in pickle_paths_set if path not in test_sentinel_full_paths]

    train_set_sentinel, valid_set_sentinel, test = train_valid_test_sentinel(sentinel_set_no_repitition,
                                                                             train_ratio=0.8227, val_ratio=0.1773,
                                                                             test_ratio=0.0)
    train_set_sentinel.extend(test)

    # Load data from the pickle file
    with open(data_path_landsat, "rb") as f:
        data = pickle.load(f)

    train_set_landsat = [
        (
            image_path.replace('/media/ladmin/Vault/Splited_biome_384', './landsat_data'),
            mask_path.replace('/media/ladmin/Vault/Splited_biome_384', './landsat_data')
        )
        for image_path, mask_path in data["train_set_landsat"]
    ]

    valid_set_landsat = [
        (
            image_path.replace('/media/ladmin/Vault/Splited_biome_384', './landsat_data'),
            mask_path.replace('/media/ladmin/Vault/Splited_biome_384', './landsat_data')
        )
        for image_path, mask_path in data["valid_set_landsat"]
    ]

    train_loader_sentinel = loader.dataloader(
        train_set_sentinel, batch_size, patch_size,
        transformations=[trf.train_base(patch_size, fixed=True),
                         trf.band_select(bands),
                         trf.normalize_to_range()
                         ],
        shuffle=True,
        num_classes=num_classes,
        num_channels=num_channels,
        left_mask_channels=num_classes)

    train_loader_landsat = loader.dataloader(
        train_set_landsat, batch_size, patch_size,
        transformations=[trf.train_base(patch_size, fixed=True),
                         trf.landsat_12_to_13(),
                         trf.class_merge(3, 4),
                         trf.class_merge(1, 2),
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
                         trf.normalize_to_range()
                         ],
        shuffle=True,
        num_classes=num_classes,
        num_channels=num_channels,
        left_mask_channels=num_classes)

    total_valid_img = len(valid_set_sentinel) + len(valid_set_landsat)
    landsat_steps = int(0.7 * total_valid_img) // batch_size
    sentinel_steps = int(0.3 * total_valid_img) // batch_size
    summary_steps = landsat_steps + sentinel_steps
    print("Total valid images: {}".format(total_valid_img))
    print(f'Summary steps: {summary_steps}')

    train_gen_sentinel = train_loader_sentinel()
    valid_gen_sentinel = valid_loader_sentinel()
    train_gen_landsat = train_loader_landsat()
    valid_gen_landsat = valid_loader_landsat()

    mixed_gen_train = loader.combined_generator(train_gen_sentinel, train_gen_landsat, sentinel_weight=0.3, landsat_weight=0.7)
    mixed_gen_valid = loader.combined_generator(valid_gen_sentinel, valid_gen_landsat, sentinel_weight=0.3, landsat_weight=0.7)

    csv_logger_save_root = os.path.join(
        os.path.dirname(model_save_path), 'training_log.csv'
    )
    model_checkpoint_save_root = os.path.join(
        os.path.dirname(model_save_path), 'model_epoch_{epoch:02d}_val_loss_{val_loss:.2f}_fine.keras'
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

    model = load_model(model_load_path)

    for layer in model.layers:
        if layer.name in ['fmm', 'multiscale_layer', 'pad_by_up', 'pad_by_up_1', 'pad_by_up_2', 'dropout', 'activation', 'input_layer']:
            layer.trainable = False
        else:
            layer.trainable = True

    for i, layer in enumerate(model.layers):
        print(f"Layer {i}: {layer.name}, Trainable: {layer.trainable}")

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
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
            f'training_history_{model_name}_{patch_size}_{epochs}_{steps_per_epoch}_fine_tuning.json'
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




