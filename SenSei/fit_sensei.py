import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.optimizers import Adagrad, SGD, RMSprop
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.python.framework.ops import disable_eager_execution

import yaml

# project_path = "/home/ladmin/PycharmProjects/cloudFCN-master"
project_path = "/mnt/agent/system/working_dir"
sys.path.append(project_path)

from SenSei.sensei_loader import SEnSeITrainer
from SenSei import spectral_encoders as encs


if __name__=='__main__':

    CONFIG = sys.argv[1]
    OUT_DIR = sys.argv[2]
    os.makedirs(OUT_DIR,exist_ok=True)

    with open(CONFIG,'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    sensei = encs.SEnSeI(config)
    if config.get('SUMMED_BAND_BLOCK'):
        recovery_module = encs.SEnSeIRecoveryModule(config['descriptor_size'],config['SUMMED_BAND_BLOCK']['final_size'])
    elif config.get('BAND_MULTIPLICATION_BLOCK'):
        recovery_module = encs.SEnSeIRecoveryModule(config['descriptor_size'],config['COMBINED_DESCRIPTOR_BLOCK']['final_size'])
    elif config.get('BAND_INTEGRATION_BLOCK'):
        recovery_module = encs.SEnSeIRecoveryModule(config['descriptor_size'],config['BAND_INTEGRATION_BLOCK']['final_size'])

    bandsin = Input(shape=(None,None,None,1))
    descriptorsin = Input(shape=(None,config['descriptor_size']))
    candidate_descriptorsin = Input(shape=(None,config['descriptor_size']))

    feature_map = sensei((bandsin,descriptorsin))

    # print("Feature map shape:", feature_map.shape)
    # print("Descriptors shape:", descriptorsin.shape)
    # print("Candidate descriptors shape:", candidate_descriptorsin.shape)

    candidate_preds,band_value_preds = recovery_module((feature_map,descriptorsin,candidate_descriptorsin))

    model = Model(inputs=(bandsin,descriptorsin,candidate_descriptorsin),outputs={'candidates':candidate_preds,'band_values':band_value_preds})
    model.summary(line_length=140)
    model.output_names = ['bands','candidates']

    callbacks=(
            TensorBoard(log_dir=os.path.join(OUT_DIR,'logs'),update_freq='epoch',profile_batch=0),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(OUT_DIR, '{epoch:03d}-{val_loss:.3f}.h5'),
                mode='min',
                monitor='val_loss',
                verbose=2,
                save_best_only=True,
                save_weights_only=False
            ),

            tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.25,
                    patience=2,
                    min_lr=2e-5
                    )
            )

    loader = SEnSeITrainer(config['descriptor_size'],1024,num_channels=(3,14))
    tester = SEnSeITrainer(config['descriptor_size'],4096,num_channels=13,test_mode=True)

    model.get_layer('SEnSeI').summary(line_length=140)
    model.summary(line_length=140)

    model.compile(optimizer = SGD(learning_rate=0.01,momentum=0.75),loss={'candidates':'mean_squared_error','band_values':'mean_squared_error'},loss_weights={'candidates':0.5,'band_values':1},metrics={'candidates':'binary_accuracy'})
    model.fit(loader,validation_data = tester, validation_steps=200,steps_per_epoch=1000, epochs=200, callbacks=callbacks)