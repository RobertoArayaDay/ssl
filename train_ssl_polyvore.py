"""
This scripts train a ssl model. So far we have

-- SimSiam <adapted from https://keras.io/examples/vision/simsiam/>
-- BYOL

This code use tfds to prepare and load the datasets. Our examepl is with
QuickDraw datasets, so you need the following repository

"""

import sys
import socket
import os
import json
from polyvore_outfits_loader import TripletImageLoader
import torch
from torchvision import transforms
import tensorflow as tf
import improc.augmentation as aug
import configparser
import models.sketch_simsiam as simsiam  
import models.sketch_byol as byol
import argparse
# import the dataset builder, here is an example for qd


def visualize_data_polyvore(ds) :
    _, ax = plt.subplots(5, 6)    
    for _ in range(5):
        sample_images_one, sample_images_two = next(iter(ds))
        for i in range(15):        
            ax[i % 5][(i // 5)*2].imshow(sample_images_one[i].numpy().astype("int"))
            ax[i % 5][((i) // 5)*2 + 1].imshow(sample_images_two[i].numpy().astype("int"))
            plt.axis("off")
        plt.waitforbuttonpress(1)
    plt.show()
    
    
def do_training(ssl_model_name, config_data, config_model, ssl_ds, model_dir, num_training_samples):
    
    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.03, 
                                                              decay_steps = config_model.getint('STEPS'))
    # Create an early stopping callback.
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", 
                                                      patience=5, 
                                                      restore_best_weights=True)
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                                                   filepath= os.path.join(model_dir, 'ckp', 'ckp_{epoch:03d}'),
                                                                   save_weights_only=True,                                                                   
                                                                   save_freq = 'epoch',  )
    history = None
    if ssl_model_name == 'SIMSIAM' :
        ssl_model = simsiam.SketchSimSiam(config_data, config_model)
        ssl_model.compile(optimizer=tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.9))
        history = ssl_model.fit(ssl_ds,
                                epochs=config_model.getint('EPOCHS'),
                                callbacks=[early_stopping, model_checkpoint_callback])
    if ssl_model_name == 'BYOL' :
        print('entra al if training')
        ssl_model = byol.SketchBYOL(config_data, config_model)
        ssl_model.compile(optimizer=tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.9))
        history = ssl_model.fit_byol(ssl_ds, num_training_samples,
                                     epochs = config_model.getint('EPOCHS'),
                                     ckp_dir = os.path.join(model_dir, 'ckp'))
    
    return history, ssl_model


def apply_default_augmentation(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.8)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_saturation(image, lower=0.2, upper=1.8)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.resize(image, size=(112, 112))
    image = (image - tf.constant([0.485, 0.456, 0.406])) / tf.constant([0.229, 0.224, 0.225])
    return image

#---------------------------------------------------------------------------------------
def ssl_map_func(image, daug_func):
    image = image['image']    
    return daug_func(image)
        
#---------------------------------------------------------------------------------------
VISUALIZE = False # if false, it runs training

AUTO = tf.data.AUTOTUNE
#---------------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type = str, required = True)    
    parser.add_argument('-model', type = str, required = True)  
    parser.add_argument('-gpu', type = int, required = False) # gpu = -1 set for using all gpus
    args = parser.parse_args()
    gpu_id = 0
    if not args.gpu is None :
        gpu_id = args.gpu    
    config_file = args.config
    ssl_model_name = args.model
    assert os.path.exists(config_file), '{} does not exist'.format(config_file)        
    assert ssl_model_name in ['BYOL', 'SIMSIAM'], '{} is not a valid model'.format(ssl_model_name)
    
    #load configuracion file
    config = configparser.ConfigParser()
    config.read(config_file)
    config_model = config[ssl_model_name]
    assert not config_model == None, '{} does not exist'.format(ssl_model_name)
    
    config_data = config['DATA']
    
    # retrieve information for polyvore dataset
    dataset_name = config_data.get('DATASET')
    datadir = config_data.get('DATADIR')
    
    fn = os.path.join(datadir, 'polyvore_outfits', 'polyvore_item_metadata.json')
    meta_data = json.load(open(fn, 'r'))

    train_loader = torch.utils.data.DataLoader(
        TripletImageLoader(datadir, 'nondisjoint', 'train', meta_data,
                               transform=transforms.Compose([
                               transforms.ToTensor(),
                           ])),
        batch_size=1, shuffle=True)
    
    #sample_batch = next(iter(train_loader))
    
    #img1_shape, img2_shape, condition_shape = sample_batch[0].squeeze(0).shape, sample_batch[1].squeeze(0).shape, #sample_batch[2].shape

    
    ds = tf.data.Dataset.from_generator(
        lambda: ((img1.squeeze(0), img2.squeeze(0), condition) for img1, img2, condition in train_loader),
        output_signature=(
            tf.TensorSpec((3, 300, 300), dtype=tf.float32),
            tf.TensorSpec((3, 300, 300), dtype=tf.float32),
            tf.TensorSpec((1,), dtype=tf.float32)
        )
    )
    
    
    target_image_size = (112, 112)
    ssl_ds = (
        ds.map(lambda x, y, z: (apply_default_augmentation(tf.transpose(x, perm=[1,2,0])),
                                apply_default_augmentation(tf.transpose(y, perm=[1,2,0])),
                                z))
        .shuffle(1024, seed=config_model.getint('SEED'))
        .batch(config_model.getint('BATCH_SIZE'), drop_remainder=True)
        .prefetch(AUTO)
    )
            
    
    # # Visualize a few augmented images.
    if VISUALIZE:
        print('if visualize')
        import matplotlib.pyplot as plt
        visualize_data_polyvore(ds)
    else :
        print('if not visualize')
        #----------------------------------------------------------------------------------
        model_dir =  config_model.get('MODEL_DIR')
        model_dir = os.path.join(model_dir, dataset_name, ssl_model_name)
        if not os.path.exists(model_dir) :
            os.makedirs(os.path.join(model_dir, 'ckp'))
            os.makedirs(os.path.join(model_dir, 'model'))
            print('--- {} was created'.format(os.path.dirname(model_dir)))
        #----------------------------------------------------------------------------------
        print('debugging')
        tf.debugging.set_log_device_placement(True)   
        # Create a cosine decay learning scheduler.
        num_training_samples = 686851
        steps = config_model.getint('EPOCHS') * (num_training_samples // config_model.getint('BATCH_SIZE'))
        config_model['STEPS'] = str(steps)  
        # Compile model and start training.
        
        print('antes de compilar modelo')
        if gpu_id >= 0 :
            with tf.device('/device:GPU:{}'.format(gpu_id)) :
                history, ssl_model = do_training(ssl_model_name, config_data, config_model, ssl_ds, model_dir, steps)                
        else :
            gpus = tf.config.list_logical_devices('GPU')
            strategy = tf.distribute.MirroredStrategy(gpus)
            with strategy.scope() :
                history, ssl_model = do_training(ssl_model_name, config_data, config_model, ssl_ds, model_dir, steps)
                              
        #predicting                    
        #hisitory = simsiam.evaluate(ssl_ds)
        # Visualize the training progress of the model.
        # Extract the backbone ResNet20.
        #saving model
        # print('saving model')
        model_file = os.path.join(model_dir, 'model', 'model')
        
        ssl_model.save_weights(model_file)
        print("model saved to {}".format(model_file))        
    #