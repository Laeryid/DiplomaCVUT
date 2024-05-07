import h5py
import numpy as np
import tensorflow as tf
import os
import random
import pandas as pd
import math


dataset_types_dict = {'ImprovedFourieMagnitude': ['I_F_X_m'],
                      'ImprovedFourieMagnitudePhase': ['I_F_X_m', 'I_F_X_p'],
                      'ImprovedFourieMagnitudeXY': ['I_F_X_m', 'I_F_X_p'],
                      'ClassicFourieMagnitude': ['F_X_m'],
                      'ClassicFourieMagnitudePhase': ['F_X_m', 'F_X_p'],
                      'ClassicFourieMagnitudeXY': ['F_X_m', 'F_X_p'],
                      'TimeDomain': ['X'],
                      'MDCT': ['mdct_X'],
                      'WT': ['WT_X']
                }

def load_file_to_numpy(file: str):    
    hf = h5py.File(file,'r')
    source = list(hf.keys())[0]
    data = np.array(hf.get(f'{source}')) 
    hf.close()
    return data
    
def load_file(file):
    #source = re.findall(r'[^/]+\.hdf5', file.numpy().decode("utf-8") )[-1].replace('.hdf5', '')
    hf = h5py.File(file.numpy().decode("utf-8"),'r')
    source = list(hf.keys())[0]
    data = np.array(hf.get(f'{source}')) 
    hf.close()
    return data

def load_file_wrapper(file):
    e = tf.py_function(load_file, [file], (tf.float64))
    return e

def create_dataset(source, filepath, source_number = np.nan):
    if source_number == source_number:
        end_string = f'_{source_number}.hdf5'
    else:
        end_string = '.hdf5'
    #data = np.array([row for f in sorted(os.listdir(filepath), key=lambda x: x) if f.startswith(source) and f.endswith(end_string) 
    #            for row in load_file_wrapper(filepath + '/' + f)])
    #dataset = tf.data.Dataset.from_tensor_slices(data)
    data = []
    for file in [filepath + '/' + f for f in sorted(os.listdir(filepath), key=lambda x: x) if f.startswith(source) and f.endswith(end_string)]:
        data.append(load_file_to_numpy(file))
    return np.concatenate(data)

def read_and_random_undersampling_dataset(file_path
                                          , type
                                          , file_num = np.nan
                                          , dataset_type = 'ImprovedFourieMagnitude'
                                          , undersampling = True
                                          , undersampling_ratio = 1
                                          , return_numpy_copy = False
                                          , X_post_processing = None
                                          , Y_post_processing = None
                                         ):
    filepath = file_path + '/' + type
    if file_num == file_num:
        end_string = f'_{file_num}.hdf5'
    else:
        end_string = '.hdf5'
        
    Y_source = f'{type}_Y_'    
    Y_array = [int(f.replace('.hdf5', '').replace(Y_source, '')) for f in sorted(os.listdir(filepath), key=lambda x: x) if f.startswith(Y_source) and f.endswith(end_string)]
    
    templates_X = dataset_types_dict[dataset_type]
    X_source = []
    X_array = []
    X_source.append(f'{type}_{dataset_types_dict[dataset_type][0]}_')
    X_array.append([int(f.replace('.hdf5', '').replace(X_source[0], '')) for f in sorted(os.listdir(filepath), key=lambda x: x) if f.startswith(X_source[0]) and f.endswith(end_string)])
    if len(templates_X) == 2:
        X_source.append(f'{type}_{dataset_types_dict[dataset_type][1]}_')
        X_array.append([int(f.replace('.hdf5', '').replace(X_source[-1], '')) for f in sorted(os.listdir(filepath), key=lambda x: x) if f.startswith(X_source[-1]) and f.endswith(end_string)])
        X_array = np.intersect1d(X_array[0],X_array[1])
        
    numbers = np.intersect1d(Y_array,X_array)
    
    for num in numbers:
        Y_i = load_file_to_numpy(f"{filepath}/{Y_source}{num}.hdf5")
        Y_i_class = np.max(Y_i, axis=1)
        ind_1 = np.arange(Y_i_class.shape[0])[Y_i_class == 1]
        ind_0 = np.arange(Y_i_class.shape[0])[Y_i_class == 0]
        if undersampling:
            ind_0_u = np.random.choice(ind_0, ind_1.shape[0]*undersampling_ratio, replace=False)
        else:               
            ind_0_u = ind_0
            
        indices = np.sort(np.concatenate([ind_1, ind_0_u]))
        try:
            Y = np.concatenate([Y, Y_i[indices]])
        except NameError:
            Y = Y_i[indices]
            
        X_i = load_file_to_numpy(f"{filepath}/{X_source[0]}{num}.hdf5")
        try:
            X_m = np.concatenate([X_m, X_i[indices]])
        except NameError:
            X_m = X_i[indices]
            
        if len(templates_X) == 2:
            X_i = load_file_to_numpy(f"{filepath}/{X_source[1]}{num}.hdf5") / 10.0
            try:
                X_p = np.concatenate([X_p, X_i[indices]])
            except NameError:
                X_p = X_i[indices]
        del X_i
        del Y_i
    
    if dataset_type[-5:] == 'Phase':
        X_m = np.concatenate([X_m, X_p], axis=-1)
    if dataset_type[-2:] == 'XY':
        X_X = X_m * np.cos(X_p)
        X_Y = X_m * np.sin(X_p)
        X_m = np.concatenate([X_m, X_X, X_Y], axis=-1)
    
    if not Y_post_processing is None:
        Y = np.array([Y_post_processing['function'](r) for r in Y])
    if not X_post_processing is None:
        X_m = np.array([X_post_processing['function'](r) for r in X_m])
        
    print(f"Shapes: {X_m.shape} and {Y.shape}")  
    
    if return_numpy_copy:
        return X_m, Y
        
    dataset_y = tf.data.Dataset.from_tensor_slices(Y.astype(float))
    dataset_X = tf.data.Dataset.from_tensor_slices(X_m.astype(float))
    dataset = tf.data.Dataset.zip((dataset_X, dataset_y))
    
    del X_m
    del Y
       
    return dataset

class hdf5_file_dataset(tf.keras.utils.Sequence):
    
    def __init__(self
                 , file_path
                 , file_type
                 , batch_size = 8
                 , dataset_type = 'ImprovedFourieMagnitude'
                 , file_num = None
                 , undersampling = False
                 , shuffle = False
                 , one_disease_undersampling = -1
                 , one_disease_dataset = -1
                 , max_disease_count = -1
                 , X_post_processing = None
                 , Y_post_processing = None
                ):
        # Initialization
        self.batch_size = batch_size
        self.dataset_type = dataset_type
        self.templates_X = dataset_types_dict[self.dataset_type]
        self.filepath = file_path + '/' + file_type
        self.file_type = file_type
        self.file_num = file_num
        self.shuffle = shuffle
        self.undersampling = undersampling
        self.one_disease_undersampling = one_disease_undersampling
        self.max_disease_count = max_disease_count
        self.one_disease_dataset = one_disease_dataset
        self.X_post_processing = X_post_processing
        self.Y_post_processing = Y_post_processing
        
        if self.one_disease_dataset != -1:
            self.Y_post_processing = {'function': lambda x: [x[self.one_disease_dataset]], 
                                      'shape': tuple([1])}
            
        self.prepare_file_numbers()
        self.prepare_metadata()
        self.on_epoch_end()
        
    def prepare_file_numbers(self):
        if self.file_num is None:
            end_string = ['.hdf5']
        if type(self.file_num).__name__ == 'int':            
            end_string = [f'_{self.file_num}.hdf5']
        if type(self.file_num).__name__ in ('list', 'tuple'):
            end_string = [f'_{x}.hdf5' for x in self.file_num]
        
        self.Y_source = f'{self.file_type}_Y_' 
        Y_array = [int(f.replace('.hdf5', '').replace(self.Y_source, '')) for e_s in end_string for f in sorted(os.listdir(self.filepath), key=lambda x: x) if f.startswith(self.Y_source) and f.endswith(e_s)]
        
        templates_X = dataset_types_dict[self.dataset_type]
        self.X_source = f'{self.file_type}_{dataset_types_dict[self.dataset_type][0]}_'
        X_array = [int(f.replace('.hdf5', '').replace(self.X_source, '')) for e_s in end_string for f in sorted(os.listdir(self.filepath), key=lambda x: x) if f.startswith(self.X_source) and f.endswith(e_s)]

        self.file_numbers = np.intersect1d(Y_array,X_array)
        
    def prepare_metadata(self):
        self.metadata = {}
        self.hdf5 = {}
        self.undersampled_indices = {}
   
        for num in self.file_numbers:
            num_dict = {}
            Y_i = load_file_to_numpy(f"{self.filepath}/{self.Y_source}{num}.hdf5")
            if self.max_disease_count != -1:
                Y_sum_class = np.sum(Y_i, axis=1)
                max_disease_count_condition = (Y_sum_class <= self.max_disease_count)
            else:
                max_disease_count_condition = [True]*Y_i.shape[0]
                
            if (not self.undersampling) and (self.one_disease_undersampling == -1):
                self.undersampled_indices[num] = np.arange(Y_i.shape[0])[max_disease_count_condition]
            else:            
                Y_max_class = np.max(Y_i, axis=1)
                if self.one_disease_undersampling != -1:
                    Y_i_class = Y_i[:, self.one_disease_undersampling]
                    ind_1 = np.arange(Y_i_class.shape[0])[(Y_i_class == 1)*max_disease_count_condition]
                    ind_0_0 = np.arange(Y_i_class.shape[0])[(Y_i_class == 0)*(Y_max_class == 0)*max_disease_count_condition]  
                    ind_0_1 = np.arange(Y_i_class.shape[0])[(Y_i_class == 0)*(Y_max_class == 1)*max_disease_count_condition] 
                    ind_0_0_u = np.random.choice(ind_0_0, ind_1.shape[0]//2, replace=False) 
                    ind_0_1_u = np.random.choice(ind_0_1, ind_1.shape[0]//2, replace=False)  
                    self.undersampled_indices[num] = np.sort(np.concatenate([ind_1,ind_0_0_u,ind_0_1_u]))                               
                else:
                    ind_1 = np.arange(Y_max_class.shape[0])[(Y_max_class == 1)*max_disease_count_condition]
                    ind_0 = np.arange(Y_max_class.shape[0])[(Y_max_class == 0)*max_disease_count_condition]
                    ind_0_u = np.random.choice(ind_0, ind_1.shape[0], replace=False)
                    self.undersampled_indices[num] = np.sort(np.concatenate([ind_1,ind_0_u]))
                    
            num_dict['len'] = self.undersampled_indices[num].shape[0]                        
            num_dict['Y_file_name'] = f"{self.filepath}/{self.Y_source}{num}.hdf5"
            num_dict['X_file_name'] = f"{self.filepath}/{self.X_source}{num}.hdf5"
            self.metadata[num] = num_dict
            del Y_i
            
        self.start_file_num = np.min(list(self.metadata.keys()))
        
        curr_number = 0
        self.file_array = []
        inner_index_array = []
        for num in range(np.max(list(self.metadata.keys()))+1):
            try:
                self.metadata[num]['start'] = curr_number
                curr_number += self.metadata[num]['len']
                self.metadata[num]['end'] = curr_number-1
                self.file_array = self.file_array + [num]*self.metadata[num]['len']
            except KeyError:
                continue
        
        self.Y_data_shape = self.read_shape(self.metadata[self.start_file_num]['Y_file_name'])
        self.X_data_shape = self.read_shape(self.metadata[self.start_file_num]['X_file_name'])
        
        if not self.Y_post_processing is None:
            self.Y_shape = self.Y_post_processing['shape']
        else:
            self.Y_shape = self.Y_data_shape            
            
        if not self.X_post_processing is None:
            self.X_shape = self.X_post_processing['shape']
        else:
            self.X_shape = self.X_data_shape
            
        self.length = curr_number
    
    def __len__(self):
        return math.ceil(self.length * 1.0 / self.batch_size)

    def __getitem__(self, index):
        # Generate indexes of the batch
        start_index = index*self.batch_size
        end_index = (index+1)*self.batch_size - 1
        if end_index > self.length:
            end_index = self.length
            portion_size = end_index - start_index
        else:
            portion_size = self.batch_size
        
        X = np.zeros([portion_size] + list(self.X_data_shape))
        y = np.zeros([portion_size] + list(self.Y_data_shape))     
        insert_start = 0

        # Find list of IDs
        for num, partition in self.metadata.items():
            if start_index <= partition['end'] and end_index >= partition['start']:
                start_index_i, end_index_i = start_index, end_index
                if start_index < partition['start']:
                    start_index_i = partition['start']
                if end_index > partition['end']:
                    end_index_i = partition['end']
                indices_i = self.indices.loc[start_index_i:end_index_i, 'inner_index'].sort_values().values
                undersampled_indices_i = np.sort(self.undersampled_indices[num][indices_i])
                X[insert_start:insert_start+end_index_i-start_index_i+1] = self.read_rows(partition['X_file_name'], undersampled_indices_i)
                y[insert_start:insert_start+end_index_i-start_index_i+1] = self.read_rows(partition['Y_file_name'], undersampled_indices_i)
                insert_start = insert_start+end_index_i-start_index_i+1
                #print(f'file {num}, indices {indices_i}')
        #print(y.shape)
        if not self.Y_post_processing is None:
            y = np.array([self.Y_post_processing['function'](r) for r in y])
        if not self.X_post_processing is None:
            X = np.array([self.X_post_processing['function'](r) for r in X])
        return X, y
    
    def on_epoch_end(self):
        self.current_file_num = self.start_file_num
        
        inner_index_array = []
        for num in range(np.max(list(self.metadata.keys()))+1):
            try:
                rang = list(range(self.metadata[num]['len']))
                if self.shuffle == True:
                    random.shuffle(rang)
                inner_index_array = inner_index_array + rang
            except KeyError:
                continue
        self.indices = pd.DataFrame({'file':self.file_array, 'inner_index':inner_index_array})            
        
    def create_file_link(self, file_name):
        self.hdf5[file_name] = h5py.File(file_name,'r')
        source = list(self.hdf5[file_name].keys())[0]
        return np.array(self.hdf5[file_name].get(f'{source}')) 
    
    def read_shape(self, file_name):
        with h5py.File(file_name,'r') as f:
            source = list(f.keys())[0]
            return f[source][0].shape
        
    def read_rows(self, file_name, indices_i):
        with h5py.File(file_name,'r') as f:
            source = list(f.keys())[0]
            return f[source][indices_i]
        
    def batch(self, batch_size):
        # We just ignore this function
        return self
    
    def shuffle(self, **kwargs):
        # We just ignore this function
        return self
        