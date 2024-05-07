import numpy as np
import pandas as pd
import h5py
import scipy
import math
import re
import json
import os

import logging
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        CSVLogger, EarlyStopping, TerminateOnNaN, Callback)
from csv import writer
from datetime import datetime

from load_dataset import create_dataset, read_and_random_undersampling_dataset, hdf5_file_dataset
from callback_save_files import TrainingCallback, FitCallback
import local_settings
from CNN_creation import CreateModel, form_CNN_parameters
from model_Article_CODE import get_model as CODE_model_create
from RNN_LSTM_Transformer import get_Transformer_model, get_RNN_CNN_model, get_RNN_model, get_LSTM_CNN_model, get_LSTM_model

from sklearn.metrics import precision_recall_curve

WORKING_DIRECTORY = local_settings.WORKING_DIRECTORY #'/home/buliabog/Diploma/'
logs_dir = f'{WORKING_DIRECTORY}Data/logs/'
file_name = '20240330 Sequensial train DNN'
task_name = ''

logs_directory = os.path.join(logs_dir, file_name)
metrics_file = os.path.join(logs_directory, 'f1.csv')

val_filepath = f'{WORKING_DIRECTORY}Data/CODE-15/val'
train_filepath = f'{WORKING_DIRECTORY}Data/CODE-15/train'

batch_size = 64

def CNN_4L_create(input_shape, n_classes):
    lr = 0.001
    space = { 'kernel_0' : (1, 2, 4, 6),     # kernel
          'max_pooling_0': (1, 4, 8, 12, 16),      # max_pooling
          'units_0': (2, 4, 8, 12, 16, 20, 30, 50, 80, 160), # units/filters   
          'activation_0': ('swish', 'linear'), # activation
         'kernel_1' : (1, 2, 4, 6),     # kernel
          'max_pooling_1': (1, 4, 8, 12, 16),      # max_pooling
          'units_1': (2, 4, 8, 12, 16, 20, 30, 50, 80, 160), # units/filters   
          'activation_1': ('swish', 'linear'), # activation
         'kernel_2' : (1, 2, 4, 6),     # kernel
          'max_pooling_2': (1, 4, 8, 12, 16),      # max_pooling
          'units_2': (2, 4, 8, 12, 16, 20, 30, 50, 80, 160), # units/filters   
          'activation_2': ('swish', 'linear'), # activation
         'kernel_3' : (1, 2, 4, 6),     # kernel
          'max_pooling_3': (1, 4, 8, 12, 16),      # max_pooling
          'units_3': (2, 4, 8, 12, 16, 20, 30, 50, 80, 160), # units/filters   
          'activation_3': ('swish', 'linear') # activation
        }
    spaces = list(space.values())
    best_state = [1,0,9,0,2,3,5,1,1,0,9,0,3,2,7,1]
    CNN_params = form_CNN_parameters(best_state, spaces)
    
    model = CreateModel(CNN_params, input_shape, n_classes)
    
    return model

clasifiers = [ {'name': "CODE ResNet"
                , 'clasifier': CODE_model_create
                , 'type': 'DNN'},
               {'name': "4L CNN"
                , 'clasifier': CNN_4L_create
                , 'type': 'DNN'},              
               {'name': "Transformer"
                , 'clasifier': get_Transformer_model
                , 'type': 'DNN'},
               {'name': "LSTM"
                , 'clasifier': get_LSTM_model
                , 'type': 'DNN'},
               {'name': "LSTM+CNN"
                , 'clasifier': get_LSTM_CNN_model
                , 'type': 'DNN'},
              # {'name': "RNN"
              #  , 'clasifier': get_RNN_model
              #  , 'type': 'DNN'},
              # {'name': "RNN+CNN"
              #  , 'clasifier': get_RNN_CNN_model
              #  , 'type': 'DNN'},               
              ]

import mdct

#def mdct_reshaped(x):
#    MDCT = np.nan_to_num(mdct.mdst(x), 0)
#    mx = np.max(np.abs(MDCT))
#    if mx != 0:        
#        MDCT = MDCT / mx
#    return np.reshape(MDCT, (-1, 12))

def mdct_reshaped(x):
    MDCT = np.nan_to_num(mdct.mdst(x), 0)
    MDCT = MDCT[:128, :, :].reshape(128, -1)
    mx = np.max(np.abs(MDCT))
    if mx != 0:        
        MDCT = MDCT / mx
    return MDCT

import pywt

def wavelet_transformation(x):
    DWT = []
    for i in range(x.shape[-1]):
        DWT.append(np.vstack(pywt.dwt(x[:, i], 'coif17')))
    #DWT = np.concatenate(pywt.dwt(x, 'coif17'), axis=1)
    DWT = np.vstack(DWT)
    mx = np.max(np.abs(DWT))
    if mx != 0:
        DWT = DWT / mx
    return DWT.T

domeny = [
    {'name': 'TimeDomain'
     , 'full name' : "Initial data without any transformation"
     , 'dataset_type': 'TimeDomain'
     , 'X_post_processing': None
     , 'shape_X' : (4096, 12)},
    {'name': 'rFFT'
     , 'full name' : "Fast Fourier Transformation for real values"
     , 'dataset_type': 'ClassicFourieMagnitude'
     , 'X_post_processing': None
     , 'shape_X' : (500, 12)},
    {'name': 'FFTwDW'
     , 'full name' : "Fast Fourier Transformation with Dynamic window"
     , 'dataset_type': 'ImprovedFourieMagnitude'
     , 'X_post_processing': None
     , 'shape_X' : (500, 12)},
    {'name': 'MDCT'
     , 'full name' : "Modified discrete cosine transform"
     , 'dataset_type': 'TimeDomain'
     , 'X_post_processing': {'function': mdct_reshaped, 'shape': (128, 108)}
     , 'shape_X' : (4096, 12)}, 
    {'name': 'WT'
     , 'full name' : "Wavelet transform"
     , 'dataset_type': 'TimeDomain'
     , 'X_post_processing': {'function': wavelet_transformation, 'shape': (2098, 24)}
     , 'shape_X' : (4096, 12)},
    ]

lr = 0.001
# CALLBACKS
callbacks = [TerminateOnNaN()
            , ReduceLROnPlateau(monitor='val_loss',
                                                   factor=0.1,
                                                   patience=6,
                                                   min_lr=lr / 100),
            EarlyStopping(patience=11,  # Patience should be larger than the one in ReduceLROnPlateau
                                               min_delta=0.00001)            
            ]


if not os.path.exists(logs_directory):
            # create dir
            os.mkdir(logs_directory) 
if not os.path.exists(metrics_file):
    # create metrics_file
    with open(metrics_file, 'w') as f:
        f.write( 'Clasifier;Domena;Task;Run;TrainDuration;TestType;F1;TP;FP;FN;TN\n')

print('Exploration start.')

def train_clasifier(clasifier, domena, task, start_run):  
    cl_name = clasifier['name']
    dm_name= domena['name']
    print(f'Clasifier {cl_name}, task {task}, domena {dm_name} start.')
    if start_run == 3:
        return "Was already done"

    Y_post_processing = {'function': lambda x: [x[task]], 'shape': tuple([1])}

    if dm_name in ( 'MDCT', 'WT'):
        val_dataset = hdf5_file_dataset(f'{WORKING_DIRECTORY}Data/CODE-15', 'val'
                                                              #, file_num = file_number
                                                               , dataset_type = dm_name
                                                               , undersampling = False
                                                               , batch_size = batch_size 
                                                               , Y_post_processing = Y_post_processing
                                                              )
        train_dataset = hdf5_file_dataset(f'{WORKING_DIRECTORY}Data/CODE-15', 'train_US_ratio_1'
                                                              #, file_num = file_number
                                                               , dataset_type = dm_name
                                                               , undersampling = False
                                                               , batch_size = batch_size 
                                                               , Y_post_processing = Y_post_processing
                                                              ) 
    else:
        val_dataset = hdf5_file_dataset(f'{WORKING_DIRECTORY}Data/CODE-15', 'val'
                                                              #, file_num = file_number
                                                               , dataset_type = domena['dataset_type']
                                                               , undersampling = False
                                                               , batch_size = batch_size 
                                                               , X_post_processing = domena['X_post_processing']
                                                               , Y_post_processing = Y_post_processing
                                                              )
        train_dataset = hdf5_file_dataset(f'{WORKING_DIRECTORY}Data/CODE-15', 'train_US_ratio_1'
                                                              #, file_num = file_number
                                                               , dataset_type = domena['dataset_type']
                                                               , undersampling = False
                                                               , batch_size = batch_size 
                                                               , X_post_processing = domena['X_post_processing']
                                                               , Y_post_processing = Y_post_processing
                                                              ) 

    test_dataset = hdf5_file_dataset(f'{WORKING_DIRECTORY}Data/CODE-15', 'test'
                                                              #, file_num = file_number
                                                               , dataset_type = domena['dataset_type']
                                                               , undersampling = False
                                                               , batch_size = batch_size 
                                                               , X_post_processing = domena['X_post_processing']
                                                               , Y_post_processing = Y_post_processing
                                                              )
    atest_dataset = hdf5_file_dataset(f'{WORKING_DIRECTORY}Data/CODE-15', 'atest'
                                                              #, file_num = file_number
                                                               , dataset_type = domena['dataset_type']
                                                               , undersampling = False
                                                               , batch_size = batch_size 
                                                               , X_post_processing = domena['X_post_processing']
                                                               , Y_post_processing = Y_post_processing
                                                              ) 
    
    if 'X_post_processing' in domena.keys() and domena['X_post_processing'] is not None:
        X_shape = domena['X_post_processing']['shape']
    else:
        X_shape = domena['shape_X']

    for run in range(3):
        if run < start_run:
            continue
            
        """
        if cl_name in ('RNN', 'RNN+CNN') and dm_name = 'TimeDomain':
            test_f1 = -1            
            test_th = 0   
            atest_f1 = -1
            atest_th = 0
            test_TP = 0
            test_FP = 0
            test_FN = np.sum(test_labels)
            test_TN = np.sum((1-test_labels))  
            atest_TP = 0
            atest_FP = 0
            atest_FN = np.sum(atest_labels)
            atest_TN = np.sum((1-atest_labels))
            row_key = f'{cl_name};{dm_name};{task};{run}'        
            with open(metrics_file, 'a') as f:
                f.write( f'{row_key};{duration};test;{test_f1};{test_TP};{test_FP};{test_FN};{test_TN}\n')  
                f.write( f'{row_key};{duration};atest;{atest_f1};{atest_TP};{atest_FP};{atest_FN};{atest_TN}\n') 
            return f'Run {row_key}, train_duration {0}: F1_test = {test_f1}, F1_atest = {atest_f1}'
        """    
        # Specify an invalid GPU device
        with tf.device('/device:GPU:1'):
            clf = clasifier['clasifier'](X_shape, 1)                        
            #model.summary()
            clf.compile(loss=tf.keras.losses.BinaryCrossentropy() #'binary_crossentropy'
                             , optimizer=tf.keras.optimizers.Adam(lr)
                             #, metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')
                             #       , tf.keras.metrics.Precision(name='precision')
                             #       , tf.keras.metrics.Recall(name='recall')]
                         )
            train_start_time = datetime.now()
            clf.fit(train_dataset
                    , epochs=20
                    , callbacks=callbacks
                    , validation_data = (val_dataset)
                    , verbose=1
                   )
        duration_delta = datetime.now() - train_start_time  
        # Specify an invalid GPU device
        with tf.device('/device:GPU:1'):
            test_prediction = clf.predict(test_dataset)[:, 0]
            atest_prediction = clf.predict(atest_dataset)[:, 0]  
            test_labels = np.array(np.concatenate([row[1] for row in test_dataset]))[:, 0]
            atest_labels = np.array(np.concatenate([row[1] for row in atest_dataset]))[:, 0]             
    
        duration = duration_delta.total_seconds() 
        try:
            test_precision, test_recall, test_treshold = precision_recall_curve(test_labels
                                                       , test_prediction
                                                      )
            test_f1_array = np.nan_to_num(2 * test_precision * test_recall / (test_precision + test_recall))
            test_f1 = np.max(test_f1_array)
            test_th = test_treshold[np.argmax(test_f1_array)-1]
        except:
            test_f1 = -1            
            test_th = 0    
            
        test_TP = np.sum((test_prediction > test_th)*test_labels)
        test_FP = np.sum((test_prediction > test_th)*(1-test_labels))
        test_FN = np.sum((test_prediction <= test_th)*test_labels)
        test_TN = np.sum((test_prediction <= test_th)*(1-test_labels))  
        
        try:
            atest_precision, atest_recall, atest_treshold = precision_recall_curve(atest_labels
                                                           , atest_prediction
                                                         )
            atest_f1_array = np.nan_to_num(2 * atest_precision * atest_recall / (atest_precision + atest_recall))
            atest_f1 = np.max(atest_f1_array)

            atest_th = atest_treshold[np.argmax(atest_f1_array)-1]
        except:
            atest_f1 = -1
            atest_th = 0
        
        atest_TP = np.sum((atest_prediction > atest_th)*atest_labels)
        atest_FP = np.sum((atest_prediction > atest_th)*(1-atest_labels))
        atest_FN = np.sum((atest_prediction <= atest_th)*atest_labels)
        atest_TN = np.sum((atest_prediction <= atest_th)*(1-atest_labels))

        row_key = f'{cl_name};{dm_name};{task};{run}'
        
        with open(metrics_file, 'a') as f:
            f.write( f'{row_key};{duration};test;{test_f1};{test_TP};{test_FP};{test_FN};{test_TN}\n')  
            f.write( f'{row_key};{duration};atest;{atest_f1};{atest_TP};{atest_FP};{atest_FN};{atest_TN}\n') 
    
        del clf
        
        del test_prediction
        del atest_prediction
        del test_labels
        del atest_labels   
        print(f'Run {row_key}, train_duration {duration}: F1_test = {test_f1}, F1_atest = {atest_f1}')
        
    del train_dataset
    del val_dataset
    del test_dataset
    del atest_dataset        

    return f'Run {row_key}, train_duration {duration}: F1_test = {test_f1}, F1_atest = {atest_f1}'

# In[22]:

def main():    
    with open(metrics_file, 'r') as f:
        rows_processed = f.read().split('\n')
        
    rows_processed_keys = [';'.join(r.split(';')[:4]) for r in rows_processed]
        
    tf.debugging.set_log_device_placement(True)
   
    for domena in domeny:
        for task in [1,2,3]:  
            # iterate over classifierss
            for clasifier in clasifiers:
                cl_name = clasifier['name']
                cl_type = clasifier['type']
                dm_name= domena['name']
                start_run = 3
                for run in range(3):
                    if f'{cl_name};{dm_name};{task};{run}' not in rows_processed_keys:
                        start_run = run
                        break
                print(train_clasifier(clasifier, domena, task, start_run))


if __name__ == '__main__':
    main()
