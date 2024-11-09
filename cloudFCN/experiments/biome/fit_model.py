from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adagrad, SGD, Adadelta, Adam
import tensorflow as tf

import json
import sys
import os

project_path = "/home/ladmin/PycharmProjects/cloudFCN-master"
sys.path.append(project_path)
tf.config.threading.get_inter_op_parallelism_threads()

# OUR STUFF
from cloudFCN.data import loader, transformations as trf
from cloudFCN.data.Datasets import LandsatDataset, train_valid_test, randomly_reduce_list
from cloudFCN import models, callbacks
from cloudFCN.experiments import custom_callbacks
from MFCNN import model_mfcnn_def
from cxn import cxn_model
from cloudFCN.data.loader import load_paths


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

    if model_checkpoint_dir is not None:
        chkpnt_path = os.path.join(
            model_checkpoint_dir, 'weights_2.{epoch:02d}-{val_loss:.2f}.keras')

    if bands is not None:
        num_channels = len(bands)
    else:
        num_channels = 12

    train_set = load_paths(train_loader_path)
    summary_valid_set, valid_paths = load_paths(valid_loader_path, valid=True)
    if valid_paths:
        valid_paths = list(dict.fromkeys(os.path.dirname(path) for path in valid_paths))
    test_set = load_paths(test_loader_path)

    if not train_set:
        train_path, valid_paths, test_paths = train_valid_test(data_path,
                                                               train_ratio=0.96,
                                                               test_ratio=0,
                                                               dataset=dataset_name,
                                                               only_test=False,
                                                               no_test=True)
        summary_valid_path = valid_paths
        print("Before creating LandsatDataset objects")
        train_set = LandsatDataset(train_path, cache_file=f"cache_train_{dataset_name}_{len(train_path)}.pkl")
        summary_valid_set = LandsatDataset(summary_valid_path,
                                           cache_file=f"cache_valid_{dataset_name}_{len(summary_valid_path)}.pkl")
        summary_valid_set.randomly_reduce(summary_valid_percent)
        test_set = LandsatDataset(test_paths, cache_file=f"cache_test_{dataset_name}_{len(test_paths)}.pkl")
        print("After creating LandsatDataset objects")

    train_loader = loader.dataloader(
        train_set, batch_size, patch_size,
        transformations=[trf.train_base(patch_size),
                         trf.band_select(bands),
                         trf.class_merge(3, 4),
                         trf.class_merge(1, 2),
                         # trf.sometimes(0.1, trf.bandwise_salt_and_pepper(
                         #     0.001, 0.001, pepp_value=-3, salt_value=3)),
                         # trf.sometimes(0.2, trf.salt_and_pepper(
                         #     0.003, 0.003, pepp_value=-3, salt_value=3)),
                         # trf.sometimes(0.2, trf.salt_and_pepper(
                         #     0.003, 0.003, pepp_value=-4, salt_value=4)),
                         # trf.sometimes(0.2, trf.salt_and_pepper(
                         #     0.003, 0.003, pepp_value=-5, salt_value=5)),
                         # trf.sometimes(0.5, trf.intensity_scale(0.9, 1.1)),
                         # trf.sometimes(0.5, trf.intensity_shift(-0.05, 0.05)),
                         # trf.sometimes(0.5, trf.chromatic_scale(0.90, 1.1)),
                         # trf.sometimes(0.5, trf.chromatic_shift(-0.05, 0.05)),
                         # trf.sometimes(0.8, trf.white_noise(0.2)),
                         # trf.sometimes(0.2, trf.quantize(2**5)),
                         # trf.sometimes(0.1, trf.white_noise(0.8)),
                         trf.normalize_to_range()
                         ],
        shuffle=True,
        num_classes=num_classes,
        num_channels=num_channels,
        remove_mask_chanels=False)

    foga_valid_sets = [LandsatDataset(valid_path, save_cache=False) for valid_path in valid_paths]
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

    summary_valid_set = randomly_reduce_list(summary_valid_set, summary_valid_percent)
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

    if model_load_path:
        model = load_model(model_load_path)

    elif model_name == "cloudfcn":
        model = models.build_model5(
            batch_norm=True, num_channels=num_channels, num_classes=num_classes)
        optimizer = Adadelta()
        model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'],
                      optimizer=optimizer)
        model.summary()

    elif model_name == "mfcnn":
        model = model_mfcnn_def.build_model_mfcnn(
            num_channels=num_channels, num_classes=num_classes)
        optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)
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
    summary_valid_gen = summary_valid_loader()
    foga_valid_gens = [foga_valid_loader()
                       for foga_valid_loader in foga_valid_loaders]
    callback_list = [custom_callbacks.foga_table5_Callback_no_thin(
        foga_valid_sets, foga_valid_gens, frequency=1)

    ]
    if model_checkpoint_dir is not None:
        callback_list.append(ModelCheckpoint(chkpnt_path, monitor='val_loss', verbose=0,
                                             save_best_only=True, save_weights_only=False,
                                             save_freq='epoch'))
    history = model.fit(
        train_gen,
        validation_data=summary_valid_gen,
        validation_steps=summary_steps,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1,
        callbacks=callback_list
    )
    with open(f'training_history_{model_name}_{patch_size}_{epochs}_{steps_per_epoch}.json', 'w') as f:
        json.dump(history.history, f)

    model.save(model_save_path)
    return model


if __name__ == "__main__":
    config_path = sys.argv[1]  # TAKE COMMAND LINE ARGUMENT
    with open(config_path, 'r') as f:
        config = json.load(f)
    model = fit_model(config)
