#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import h5py
import scipy
import math
import datetime
import re
import json
import os

import tensorflow as tf
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        CSVLogger, EarlyStopping)
from csv import writer
from datetime import datetime

from load_dataset import create_dataset, read_and_random_undersampling_dataset, hdf5_file_dataset
from callback_save_files import TrainingCallback, FitCallback
import local_settings
from anneal_simulation import AnnealSimulation
from model_Article_CODE import get_model as get_CODE_model, restore_model as restore_CODE_model

WORKING_DIRECTORY = local_settings.WORKING_DIRECTORY #'/home/buliabog/Diploma/'
logs_dir = f'{WORKING_DIRECTORY}Data/logs/'
file_name = '20240218 Art CODE NN full data'
task_name = ''

logs_directory = logs_dir + file_name

val_filepath = f'{WORKING_DIRECTORY}Data/CODE-15/val'
train_filepath = f'{WORKING_DIRECTORY}Data/CODE-15/train'




# # CNN

# In[2]:


# # CNN

lr = 0.001
batch_size = 32
dataset_type = 'TimeDomain'
val_dataset = hdf5_file_dataset(f'{WORKING_DIRECTORY}Data/CODE-15', 'val'
                                                      , batch_size = batch_size
                                                      #, file_num = 2
                                                      , dataset_type = dataset_type)
test_dataset = hdf5_file_dataset(f'{WORKING_DIRECTORY}Data/CODE-15', 'test'
                                                      , batch_size = batch_size
                                                      #, file_num = 2
                                                      , dataset_type = dataset_type)
atest_dataset = hdf5_file_dataset(f'{WORKING_DIRECTORY}Data/CODE-15', 'atest'
                                                      , batch_size = batch_size
                                                      #, file_num = 2
                                                      , dataset_type = dataset_type)

print(f'Val shape - X {[val_dataset.length] + list(val_dataset.X_shape)}, Y {[val_dataset.length] + list(val_dataset.Y_shape)}')
print(f'Test shape - X {[test_dataset.length] + list(test_dataset.X_shape)}, Y {[test_dataset.length] + list(test_dataset.Y_shape)}')
print(f'ATest shape - X {[atest_dataset.length] + list(atest_dataset.X_shape)}, Y {[atest_dataset.length] + list(atest_dataset.Y_shape)}')


# In[ ]:


#model_dict = {'CODE_model': get_CODE_model, 'restore_CODE_model': restore_CODE_model}
model_dict = {'CODE_model': get_CODE_model}

for run in range(3):
    for US in (True, False):
        for max_disease_count in (1, -1):
            train_dataset = hdf5_file_dataset(f'{WORKING_DIRECTORY}Data/CODE-15', 'train'
                                              , batch_size = batch_size
                                              #, file_num = 2
                                              , dataset_type = dataset_type
                                              , undersampling = US
                                              , max_disease_count = max_disease_count
                                              , shuffle = True)
            print(f'Train shape - X {[train_dataset.length] + list(train_dataset.X_shape)}, Y {[train_dataset.length] + list(train_dataset.Y_shape)}')
        
            for model_name, model_function in model_dict.items():            
                task_name = f'run {run}, US {US}, max_disease_count {max_disease_count}'              
    
                # CALLBACKS
                callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                                                   factor=0.1,
                                                                   patience=6,
                                                                   min_lr=lr / 100),
                                                 EarlyStopping(patience=11,  # Patience should be larger than the one in ReduceLROnPlateau
                                                               min_delta=0.00001),
                                                 FitCallback(logs_dir, file_name, val_dataset
                                                             , task_name = task_name
                                                             , test_data = test_dataset
                                                             , atest_data = atest_dataset)]
                
                
            
            
            
                model = model_function(6)
            
                model.summary()
            
                model.compile(loss=tf.keras.losses.BinaryCrossentropy() #'binary_crossentropy'
                                 , optimizer=tf.keras.optimizers.Adam(lr)
                                 , metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')
                                        , tf.keras.metrics.Precision(name='precision')
                                        , tf.keras.metrics.Recall(name='recall')
                                        #, tf.keras.metrics.F1Score(name='f1_score', average='micro')
                                           ])
                model.fit( train_dataset,
                                epochs=70,
                                initial_epoch=0,
                                callbacks = callbacks, 
                                validation_data=(val_dataset),
                                verbose=1)
               

