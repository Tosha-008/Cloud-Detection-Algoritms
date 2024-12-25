from tensorflow.keras.optimizers import Adadelta, Adam
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

import json
import pickle
import sys
import os

project_path = "/mnt/agent/system/working_dir"
sys.path.append(project_path)
tf.config.threading.get_inter_op_parallelism_threads()

# OUR STUFF
from data import loader, transformations as trf
from data.Datasets import train_valid_test_sentinel
from MFCNN import model_mfcnn_def
from cxn import cxn_model
from output.defs_for_output import img_mask_pair


def fit_model_sentinel(config):
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

    model_save_path = os.path.join(model_save_path,
                                   f'model_sentinel_{model_name}_{patch_size}_{epochs}_{steps_per_epoch}.keras')


    if bands is not None:
        num_channels = len(bands)
    else:
        num_channels = 13

    sentinel_set = img_mask_pair(sentinel_img_dir, sentinel_mask_dir)
    train_set, valid_set, test = train_valid_test_sentinel(sentinel_set, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    train_loader = loader.dataloader(
        train_set, batch_size, patch_size,
        transformations=[trf.train_base(patch_size, fixed=True),
                         trf.band_select(bands),
                         trf.normalize_to_range()
                         ],
        shuffle=True,
        num_classes=num_classes,
        num_channels=num_channels,
        left_mask_channels=num_classes)


    valid_loader = loader.dataloader(
        valid_set, batch_size, patch_size,
        transformations=[trf.train_base(patch_size, fixed=True),
                         trf.band_select(bands),
                         trf.normalize_to_range()
                         ],
        shuffle=False,
        num_classes=num_classes,
        num_channels=num_channels,
        left_mask_channels=num_classes)
    summary_steps = len(valid_set) // batch_size

    if model_name == "mfcnn":
        model = model_mfcnn_def.build_model_mfcnn(
            num_channels=num_channels, num_classes=num_classes, dropout_p=0.5)
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'],
                      optimizer=optimizer)
        model.summary()

    elif model_name == "cxn":
        model = cxn_model.model_arch(input_rows=patch_size, input_cols=patch_size, num_of_channels=num_channels,
                                     num_of_classes=num_classes)
        optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)
        model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'],
                      optimizer=optimizer)
        model.summary()

    else:
        raise ValueError('Choose correct model`s name')

    train_gen = train_loader()
    valid_gen = valid_loader()

    csv_logger_save_root = os.path.join(
        os.path.dirname(model_save_path), 'training_log.csv'
    )
    model_checkpoint_save_root = os.path.join(
        os.path.dirname(model_save_path), 'model_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.keras'
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

    # Combine callbacks
    callbacks = [model_checkpoint, csv_logger]

    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        validation_steps=summary_steps,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks)

    try:
        history_path = os.path.join(
            os.path.dirname(model_save_path),
            f'training_history_sentinel_{model_name}_{patch_size}_{epochs}_{steps_per_epoch}.json'
        )
        test_set_path = os.path.join(
            os.path.dirname(model_save_path),
            f'test_set{model_name}_{patch_size}_{epochs}_{steps_per_epoch}.pkl'
        )
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump(history.history, f)
        with open(test_set_path, 'wb') as f:
            pickle.dump(test, f)

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
    model = fit_model_sentinel(config)
