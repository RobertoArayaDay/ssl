"""
Thi scripst tests a ssl model using similarity search
Models :
-- SimSiam
-- BYOL
"""
import sys
import socket
import os

import json
from polyvore_outfits_loader import TripletImageLoader
import torch
from torchvision import transforms

import tensorflow as tf
import numpy as np
import models.sketch_simsiam as simsiam
import models.sketch_byol as byol
import os 
import configparser
import skimage.io as io
import tensorflow_datasets as tfds
import argparse
#---------- dataset builder --------------------  

def ssl_map_func(image, crop_size):
    image = image['image'] 
    image = tf.image.grayscale_to_rgb(image) 
    image = tf.image.resize(image, (crop_size,crop_size))
    return image
        
def imagenet_map_func(image, crop_size):
    image = image['image']    
    #image = tf.image.grayscale_to_rgb(image) 
    size = int(crop_size * 1.15)
    image = tf.image.resize_with_pad(image, size, size)
    image = tf.image.random_crop(image, size = [crop_size, crop_size, 3])
    image = tf.cast(image, tf.uint8) 
    return image    
        
class PolyvoreTest():
    def __init__(self, configfile, model):
        print('se inicializa el test')
        config = configparser.ConfigParser()
        config.read(configfile)
        self.config_model = config[model]
        self.config_data = config['DATA']
        self.model = None        
        if  model  == 'SIMSIAM' :
            ssl_model = simsiam.SketchSimSiam(self.config_data, self.config_model)
            ssl_model.load_weights(self.config_model.get('CKP_FILE')).expect_partial()        
            self.model= ssl_model.encoder
        if  model  == 'BYOL' :
            ssl_model = byol.SketchBYOL(self.config_data, self.config_model)
            ssl_model.load_weights(self.config_model.get('CKP_FILE')).expect_partial()
            self.model= ssl_model.online_encoder
        assert not (self.model == None), '-- there is not a ssl model'
        print('Model loaded OK', flush = True)            

    def load_data(self):
        datadir = self.config_data.get('DATADIR')
        fn = os.path.join(datadir, 'polyvore_outfits', 'polyvore_item_metadata.json')
        meta_data = json.load(open(fn, 'r'))
           
        
        test_loader = torch.utils.data.DataLoader(
            TripletImageLoader(datadir, 'nondisjoint', 'test', meta_data,
                                   transform=transforms.Compose([
                                   transforms.ToTensor(),
                               ])),
            batch_size=1, shuffle=True)
        
        self.test_loader = test_loader
        
        print('antes del test loader')
        ds_test = tf.data.Dataset.from_generator(
        lambda: (img1.squeeze(0) for img1 in test_loader),
        output_signature=(
            tf.TensorSpec((3, 300, 300), dtype=tf.float32)
            )
        )

        target_image_size = (112, 112)
        ds_test = (
            ds_test.map(lambda x: (tf.image.resize(tf.transpose(x, perm=[1,2,0]), target_image_size)))
            # ds.map(lambda x, y, z: (ssl_map_func(x, daug.get_augmentation_fun()), ssl_map_func(y, daug.get_augmentation_fun())), num_parallel_calls=AUTO)
            .batch(1000, drop_remainder=False)
        )
        
        
        # obtains all embeddings
        contador = 0
        all_embeddings = []
        for sample in ds_test:
            print('sample shape:', tf.shape(sample))
            embeddings = self.model.predict(sample)
            print('embeddings shape:', tf.shape(embeddings))
            all_embeddings.append(embeddings)
        all_embeddings = tf.concat(all_embeddings, 0)
        
        print('all embeddings size: ', tf.shape(all_embeddings))
        self.embeddings = all_embeddings
              
    
    def compute_metrics(self):
        embeddings = self.embeddings
        test_loader = self.test_loader
        
        auc = test_loader.dataset.test_compatibility(embeddings)
        acc = test_loader.dataset.test_fitb(embeddings)
        acc_var = test_loader.dataset.test_fitb_var(embeddings)
        total = auc + acc + acc_var
        print('\n{} set: Compat AUC: {:.2f} FITB: {:.1f}\n FITB_var: {:.1f}\n'.format(
            test_loader.dataset.split,
            round(auc, 2), round(acc * 100, 1), round(acc_var * 100, 1)))

        return total

    def visualize(self, idx):
        size = self.config_data.getint('CROP_SIZE')
        n = 10
        image = np.ones((size, n*size, 3), dtype = np.uint8)*255
        i = 0
        for i , pos in enumerate(self.data[idx, :n]) :
            image[:, i * size:(i + 1) * size, :] = self.data[pos,:,:,:]
        return image
       

    def get_dataset_name(self):
        return self.config_data.get('DATASET')     

if __name__ == '__main__' :
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
    if gpu_id >= 0 :
        with tf.device('/device:GPU:{}'.format(gpu_id)) :
            print('entra al tf device')
            polyvore_test = PolyvoreTest(config_file, ssl_model_name)
            polyvore_test.load_data()
            
            # score of the metrics
            score = polyvore_test.compute_metrics()
            
            idxs = np.random.randint(1000, size = 10)
            dataset_name = polyvore_test.get_dataset_name()
            result_dir = os.path.join('results', dataset_name, ssl_model_name)
            if not os.path.exists(result_dir) :
                os.makedirs(result_dir)
                
            for idx in idxs :
                rimage =  polyvore_test.visualize(idx)
                fname = 'result_{}_{}_{}.png'.format(dataset_name, ssl_model_name, idx)
                fname = os.path.join(result_dir,fname)
                io.imsave(fname, rimage)
                print('result saved at {}'.format(fname))
    