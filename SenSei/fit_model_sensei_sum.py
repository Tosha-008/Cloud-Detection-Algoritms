import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import Input
from tensorflow.keras.metrics import AUC, MeanIoU
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow_addons.layers import GroupNormalization
import yaml

from sensei.utils import OneHotMeanIoU
from sensei.data.loader import Dataloader, CommonBandsDataloader
from sensei.data.utils import SYNTHETIC_DICT
from sensei.data import transformations as trf
from sensei.callbacks import LearningRateLogger, ImageCallback
from sensei import models
from sensei.layers import Flatten_2D_Op, PermuteDescriptors, Tile_bands_to_descriptor_count, Concatenate_bands_with_descriptors
from sensei.deeplabv3p import Deeplabv3
import pickle


#Unknown issue means memory growth must be True, otherwise breaks. May be hardware-specific?
tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices('GPU')[0], True
)

#Mixed precision training, allows ~doubling of batch size
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

def fit_sensei(config):
    """
    Main function to train the SEnSeI model.

    Parameters:
        config (dict): Dictionary containing model settings, data paths, and hyperparameters.

    Returns:
        train_loader, valid_loader, display_loader: Data loaders for training, validation, and display.
    """

    # Extracting necessary options from the config file
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

    # Define model save path
    model_save_path = os.path.join(model_save_path,
                                   f'model_{model_name}_{epochs}_{steps_per_epoch}_commonmodel_lowclouds.keras')

    # Load the SEnSeI model or define the number of input channels based on the dataset
    if model_name == 'sensei':
        BANDS = (3,14)  # samples random set of bands numbering between 3-14
        bandsin = Input(shape=(None, None, None, 1))
        descriptorsin = Input(shape=(None, 3))

        # Load the pretrained model with custom layers
        sensei_pretrained = load_model(
                    model_load_path,
                    custom_objects={        #annoying requirement when using custom layers
                        'GroupNormalization':GroupNormalization,
                        'Flatten_2D_Op':Flatten_2D_Op,
                        'PermuteDescriptors':PermuteDescriptors,
                        'Tile_bands_to_descriptor_count':Tile_bands_to_descriptor_count,
                        'Concatenate_bands_with_descriptors':Concatenate_bands_with_descriptors
                        })
        sensei = sensei_pretrained.get_layer('SEnSeI')
        feature_map = sensei((bandsin,descriptorsin))
        NUM_CHANNELS = sensei.output_shape[-1]
    else:
        if config['S2_L8_COMMON']:
            NUM_CHANNELS=8
        else:
            NUM_CHANNELS=13

    # Define the DeepLabv3 segmentation model
    if config['MODEL_TYPE']=='DeepLabv3':
        main_model = Deeplabv3(
                        input_shape=(config['PATCH_SIZE'], config['PATCH_SIZE'], NUM_CHANNELS),
                        classes=config['CLASSES'],
                        backbone='mobilenetv2'
                        )
    else:
        print('MODEL_TYPE not recognised')
        sys.exit()

    # Define final model architecture
    if model_name == 'sensei':
        outs = main_model(feature_map)
        model = Model(inputs=(bandsin,descriptorsin), outputs=outs)
    else:
        model = main_model

    # Load dataset paths from pickle files
    with open(sentinel_paths, "rb") as f:
        sentinel_set = pickle.load(f)

    with open(data_path_landsat, "rb") as f:
        landsat_set = pickle.load(f)

    # Adjust dataset paths for Sentinel-2 images
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

    # Adjust dataset paths for Landsat images
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

    # Define data loaders for Sentinel and Landsat datasets with transformations
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

    initial_epoch = 0

    # Random validation images for tensorboard visualisation
    np.random.seed(13)
    disp_idxs = np.array(np.random.choice(np.arange(len(display_loader)),25))
    file_writer_images = tf.summary.create_file_writer(logdir + '/images')
    image_callback = ImageCallback(display_loader,file_writer_images,idxs = disp_idxs)

    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.25,
        patience=3,
        min_lr=2e-5
        )

    if config['CLASSES']==2:
        metrics=['categorical_accuracy', AUC()]

    if config['CLASSES']==3:
        metrics=['categorical_accuracy', OneHotMeanIoU(config['CLASSES'])]

    optimizer = SGD(lr=config['LR'], momentum=config['MOMENTUM'], nesterov=True)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=metrics
        )
    if config['SENSEI']:
        model.get_layer('SEnSeI').summary(line_length=160)
        model.get_layer('deeplabv3p').summary(line_length=160)
    model.summary(line_length=160)

    callback_list = [
        TensorBoard(log_dir=logdir, update_freq='epoch'),
        ModelCheckpoint(
            modeldir+'/{epoch:02d}-{val_loss:.2f}.h5',
            save_weights_only=False,
            save_best_only=True,
            monitor='val_loss',
            mode='min'
            ),
        ModelCheckpoint(
            modeldir+'/latest.h5',
            save_weights_only=False,
            save_best_only=False
            ),
        lr_schedule,
        LearningRateLogger(),
        image_callback
        ]

    model.fit(
        train_loader,
        validation_data=valid_loader,
        validation_steps=len(valid_loader),
        steps_per_epoch=config['STEPS_PER_EPOCH'],
        epochs=config['EPOCHS'],
        initial_epoch = initial_epoch,
        callbacks=callback_list,
        shuffle=True
        )

    train_transformations = [
        trf.Base(config['PATCH_SIZE']),
        trf.Class_merge(class_dict),
        trf.Sometimes(0.5,trf.Chromatic_scale(factor_min=0.95, factor_max=1.05)),
        trf.Sometimes(0.5,trf.Bandwise_salt_and_pepper(0.001,0.001,pepp_value=0,salt_value=1.1)),
        trf.Sometimes(0.1,trf.Salt_and_pepper(0.001,0.001,pepp_value=0,salt_value=1.1)),
        trf.Sometimes(0.05,trf.Quantize(30,min_value=-1,max_value=2)),
        trf.Sometimes(0.05,trf.Quantize(40,min_value=-1,max_value=2)),
        trf.Sometimes(0.5,trf.Chromatic_shift(shift_min=-0.05,shift_max=0.05)),
        trf.Sometimes(0.5,trf.White_noise(sigma=0.02)),
        trf.Sometimes(0.5,trf.Descriptor_scale(factor_min=0.99,factor_max=1.01))
        ]
    if config['SYNTHETIC_BANDS']:
        train_transformations.append(trf.Sometimes(0.5,trf.Synthetic_bands(SYNTHETIC_DICT,N=3,p=0.5)))

    valid_transformations = [
        trf.Base(config['PATCH_SIZE'], fixed=True),
        trf.Class_merge(class_dict)
        ]

    if config['SENSEI']:
        convert_shape = True
        output_descriptors = True
        descriptor_style=config['DESCRIPTOR_STYLE']
        BANDS = (3,14)
    else:
        convert_shape = False
        output_descriptors = False
        descriptor_style=config['DESCRIPTOR_STYLE']
        BANDS = 'all'

    if config['S2_L8_COMMON']: #Use CommonBandsDataloader
        train_loader = CommonBandsDataloader(config['TRAIN_DIRS'], config['BATCH_SIZE'], config['PATCH_SIZE'], shuffle=True,
                                  transformations=train_transformations,
                                  band_selection=BANDS,
                                  convert_shape = convert_shape,
                                  output_descriptors = output_descriptors,
                                  descriptor_style = descriptor_style,
                                  repeated=2 # makes sure there are enough samples for 1000 steps at batch_size = 8
                                  )

        valid_loader = CommonBandsDataloader(config['VALID_DIRS'], config['BATCH_SIZE'], config['PATCH_SIZE'],
                                  transformations=valid_transformations,
                                  band_selection='all',
                                  convert_shape = convert_shape,
                                  output_descriptors = output_descriptors,
                                  descriptor_style = descriptor_style
                                  )

        display_loader = CommonBandsDataloader(config['VALID_DIRS'], 1, config['PATCH_SIZE'],
                                  transformations=valid_transformations,
                                  band_selection='all',
                                  convert_shape = convert_shape,
                                  output_descriptors = output_descriptors,
                                  descriptor_style = descriptor_style
                                  )

    else:
        train_loader = Dataloader(config['TRAIN_DIRS'], config['BATCH_SIZE'], config['PATCH_SIZE'], shuffle=True,
                                  transformations=train_transformations,
                                  band_selection=BANDS,
                                  convert_shape = convert_shape,
                                  output_descriptors = output_descriptors,
                                  descriptor_style = descriptor_style,
                                  repeated=2 # makes sure there are enough samples for 1000 steps at batch_size = 8
                                  )

        valid_loader = Dataloader(config['VALID_DIRS'], config['BATCH_SIZE'], config['PATCH_SIZE'],
                                  transformations=valid_transformations,
                                  band_selection='all',
                                  convert_shape = convert_shape,
                                  output_descriptors = output_descriptors,
                                  descriptor_style = descriptor_style
                                  )

        display_loader = Dataloader(config['VALID_DIRS'], 1, config['PATCH_SIZE'],
                                  transformations=valid_transformations,
                                  band_selection='all',
                                  convert_shape = convert_shape,
                                  output_descriptors = output_descriptors,
                                  descriptor_style = descriptor_style
                                  )

    return (train_loader, valid_loader, display_loader)

if __name__=='__main__':
    with open(sys.argv[1],'r') as f:
        config = yaml.load(f)
    main(config)