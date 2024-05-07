import tensorflow as tf
from datetime import datetime
import numpy as np
import pandas as pd
import os


# define your custom callback for prediction
class TrainingCallback(tf.keras.callbacks.Callback): 
    def __init__(self, logs_dir, file_name, validation_data, task_name = ''):
        super().__init__()
        self.logs_dir = os.path.join(logs_dir, file_name)
        self.metrics_file = os.path.join(self.logs_dir, 'metrics.csv')
        self.sep_metrics_file = os.path.join(self.logs_dir, 'sep_metrics.csv')
        self.detailed_file = os.path.join(self.logs_dir, 'details.csv')
        self.validation_data = validation_data  # (val_x, val_y)
        self.file_name = file_name
        self.task_name = task_name
        print(validation_data)
    
    def on_train_begin(self, logs=None):
        self.start_time = datetime.now().strftime("%Y%m%d %H%M%S")
        self.predicts = []
        if not os.path.exists(self.logs_dir):
            # create dir
            os.mkdir(self.logs_dir) 
        if not os.path.exists(self.metrics_file):
            # create metrics_file
            with open(self.metrics_file, 'w') as f:
                f.write( 'Pandas_index;File_name;start_time;epoch;epoch_duration;val_loss;val_binary_accuracy;val_precision;val_recall;val_f1;lr\n')
        if not os.path.exists(self.sep_metrics_file):
            # create sep_metrics_file
            with open(self.sep_metrics_file, 'w') as f:
                f.write( 'pandas_index;file_name;start_time;epoch;label_index;val_binary_accuracy;val_precision;val_recall;val_f1\n')
        if not os.path.exists(self.detailed_file):
            # create detailed_file
            with open(self.detailed_file, 'w') as f:
                f.write('pandas_index;file_name;start_time;epoch;instance;label_index;predikce;label\n')   
        
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_time = datetime.now()        
        
    def on_epoch_end(self, epoch, logs={}):
        
        prediction = self.model.predict(self.validation_data.batch(8))
        labels = np.zeros(prediction.shape)
        name = self.file_name + (' - ' + self.task_name if self.task_name != '' else '')
        
        for instance, row in enumerate(iter(self.validation_data)):
            for label_index in range(prediction.shape[1]):
                radek = {}
                radek['file_name'] = name
                radek['start_time'] = self.start_time
                radek['epoch'] = epoch 
                radek['instance'] = instance 
                radek['label_index'] = label_index                 
                radek['predikce'] = prediction[instance, label_index]  
                radek['label'] = row[1][label_index].numpy()         
                pd.DataFrame(radek, index=[0]).to_csv(self.detailed_file, mode='a', header = False, sep=';')
            labels[instance, :] = row[1].numpy()
        
        true_positive = prediction * labels
        true_negative = (prediction - 1) * (labels - 1)
        accuracy = np.sum(true_positive + true_negative, axis = 0) / prediction.shape[0]
        tr = np.sum(true_positive, axis = 0)
        precision = tr / np.sum(prediction, axis = 0)
        recall = tr / np.sum(labels, axis = 0)
        f1 = 2*precision*recall/(precision + recall)
        
        for ind in range(prediction.shape[-1]):
            separate_metrics_list = {}
            separate_metrics_list['file_name'] = name
            separate_metrics_list['start_time'] = self.start_time
            separate_metrics_list['epoch'] = epoch
            separate_metrics_list['label_index'] = ind
            separate_metrics_list['val_binary_accuracy'] = accuracy[ind]
            separate_metrics_list['val_precision'] = precision[ind]
            separate_metrics_list['val_recall'] = recall[ind]
            separate_metrics_list['val_f1'] = f1[ind]
            pd.DataFrame(separate_metrics_list, index=[0]).to_csv(self.sep_metrics_file, mode='a', header = False, sep=';')       
        
        metrics_list = {}
        metrics_list['file_name'] = name
        metrics_list['start_time'] = self.start_time
        metrics_list['epoch'] = epoch
        metrics_list['epoch_duration'] = (datetime.now() - self.epoch_start_time).total_seconds()
        metrics_list['val_loss'] = logs['val_loss']
        metrics_list['val_binary_accuracy'] = logs['val_binary_accuracy']
        metrics_list['val_precision'] = logs['val_precision']
        metrics_list['val_recall'] = logs['val_recall']
        metrics_list['val_f1'] =  logs['val_f1_score'] #2 * logs['val_recall'] * logs['val_precision'] / (logs['val_recall'] + logs['val_precision'])
        metrics_list['lr'] = logs['lr']
        pd.DataFrame(metrics_list, index=[0]).to_csv(self.metrics_file, mode='a', header = False, sep=';')
        
# define your custom callback for prediction
class FitCallback(tf.keras.callbacks.Callback): 
    def __init__(self
                 , logs_dir
                 , file_name
                 , validation_data
                 , task_name = ''
                 , task_name_header = 'Task_name'
                 , test_data = None
                 , atest_data = None
                 , save_model = False
                 ):
        super().__init__()
        self.logs_dir = os.path.join(logs_dir, file_name)
        self.metrics_file = os.path.join(self.logs_dir, 'metrics.csv')
        self.ROC_file = os.path.join(self.logs_dir, 'ROC.csv')
        self.metrics_sep_file = os.path.join(self.logs_dir, 'metrics_sep.csv')
        self.ROC_sep_file = os.path.join(self.logs_dir, 'ROC_sep.csv')
        self.ROC_test_file = os.path.join(self.logs_dir, 'ROC_test.csv')
        self.ROC_atest_file = os.path.join(self.logs_dir, 'ROC_atest.csv')
        self.test_file = os.path.join(self.logs_dir, 'test.csv')
        self.atest_file = os.path.join(self.logs_dir, 'atest.csv')
        self.test_sep_file = os.path.join(self.logs_dir, 'test_sep.csv')
        self.atest_sep_file = os.path.join(self.logs_dir, 'atest_sep.csv')
        self.test_sep_tau_file = os.path.join(self.logs_dir, 'test_sep_tau.csv')
        self.atest_sep_tau_file = os.path.join(self.logs_dir, 'atest_sep_tau.csv')
        self.validation_data = validation_data  # (val_x, val_y)
        self.test_data = test_data  
        self.atest_data = atest_data  
        self.file_name = file_name
        self.task_name = task_name
        self.task_name_header = task_name_header
        self.save_model = save_model
    
    def on_train_begin(self, logs=None):
        self.start_time = datetime.now()
        self.start_time_str = self.start_time.strftime("%Y%m%d %H%M%S")
        self.model_save_folder = os.path.join(self.logs_dir, f'{self.start_time_str} {self.task_name}')
        if not os.path.exists(self.logs_dir):
            # create dir
            os.mkdir(self.logs_dir) 
        if self.save_model and not os.path.exists(self.model_save_folder):
            # create dir
            os.mkdir(self.model_save_folder) 
        if not os.path.exists(self.metrics_file):
            # create metrics_file
            with open(self.metrics_file, 'w') as f:
                f.write( f'File_name;{self.task_name_header};start_time;duration;last_epoch;best_treshold;TP;TN;FP;FN\n')
        if not os.path.exists(self.ROC_file):
            with open(self.ROC_file, 'w') as f:
                f.write( f'File_name;{self.task_name_header};start_time;duration;last_epoch;real_class;predict_class;treshold;TP;TN;FP;FN\n')
        if not os.path.exists(self.metrics_sep_file):
            with open(self.metrics_sep_file, 'w') as f:
                f.write( f'File_name;{self.task_name_header};start_time;duration;last_epoch;task_id;best_treshold;TP;TN;FP;FN\n')
        if not os.path.exists(self.ROC_sep_file):
            with open(self.ROC_sep_file, 'w') as f:
                f.write( f'File_name;{self.task_name_header};start_time;duration;last_epoch;real_class;predict_class;task_id;treshold;TP;TN;FP;FN\n')
        if (not os.path.exists(self.ROC_test_file)) and not(self.test_data is None):
            with open(self.ROC_test_file, 'w') as f:
                f.write( f'File_name;{self.task_name_header};start_time;duration;last_epoch;real_class;predict_class;task_id;treshold;TP;TN;FP;FN\n')
        if (not os.path.exists(self.ROC_atest_file)) and not(self.atest_data is None):
            with open(self.ROC_atest_file, 'w') as f:
                f.write( f'File_name;{self.task_name_header};start_time;duration;last_epoch;real_class;predict_class;task_id;treshold;TP;TN;FP;FN\n')
        if not os.path.exists(self.test_file):
            with open(self.test_file, 'w') as f:
                f.write( f'File_name;{self.task_name_header};start_time;duration;last_epoch;real_class;predict_class;treshold;TP;TN;FP;FN\n')
        if not os.path.exists(self.atest_file):
            with open(self.atest_file, 'w') as f:
                f.write( f'File_name;{self.task_name_header};start_time;duration;last_epoch;real_class;predict_class;treshold;TP;TN;FP;FN\n')
        if not os.path.exists(self.test_sep_file):
            with open(self.test_sep_file, 'w') as f:
                f.write( f'File_name;{self.task_name_header};start_time;duration;last_epoch;real_class;predict_class;task_id;treshold;TP;TN;FP;FN\n')
        if not os.path.exists(self.atest_sep_file):
            with open(self.atest_sep_file, 'w') as f:
                f.write( f'File_name;{self.task_name_header};start_time;duration;last_epoch;real_class;predict_class;task_id;treshold;TP;TN;FP;FN\n')
        if not os.path.exists(self.test_sep_tau_file):
            with open(self.test_sep_tau_file, 'w') as f:
                f.write( f'File_name;{self.task_name_header};start_time;duration;last_epoch;real_class;predict_class;task_id;treshold;TP;TN;FP;FN\n')
        if not os.path.exists(self.atest_sep_tau_file):
            with open(self.atest_sep_tau_file, 'w') as f:
                f.write( f'File_name;{self.task_name_header};start_time;duration;last_epoch;real_class;predict_class;task_id;treshold;TP;TN;FP;FN\n')
        
    def on_epoch_end(self, epoch, logs={}):
        self.last_epoch = epoch
        
    def save_metrics(self, TP, TN, FP, FN):
        f1 = np.nan_to_num(np.sum(TP, axis=(1, 2))/(np.sum(TP, axis=(1, 2)) + 0.5*(np.sum(FP, axis=(1, 2))+np.sum(FN, axis=(1, 2)))))
        i_best = np.argmax(f1)
        f1_sep = np.nan_to_num(np.sum(TP, axis=1)/(np.sum(TP, axis=1) + 0.5*(np.sum(FP, axis=1)+np.sum(FN, axis=1))))
        i_best_sep = np.argmax(f1_sep, axis = 0)
        
        
        #print('f1', f1)
        #print(f'i_best = {i_best}')
        #print('f1_sep', pd.DataFrame(f1_sep).to_string())
        #print(f'i_best_sep = {i_best_sep}')
        #print('f1_sep\[i_best\] = ', f1_sep[i_best, :])
        #print('f1_sep\[i_best_sep\] = ', np.max(f1_sep, axis = 0))
        
        
        self.i_best = i_best
        self.i_best_sep = i_best_sep
        
        with open(self.metrics_file, 'a') as f:
            f.write(f"{self.file_name};{self.task_name};{self.start_time_str};{self.duration};{self.last_epoch};{i_best};{np.sum(TP[i_best, :, :])};{np.sum(TN[i_best, :, :])};{np.sum(FP[i_best, :, :])};{np.sum(FN[i_best, :, :])}\n")
            
        with open(self.metrics_sep_file, 'a') as f:
            for lab in np.arange(TP.shape[2]):
                f.write(f"{self.file_name};{self.task_name};{self.start_time_str};{self.duration};{self.last_epoch};{lab};{i_best_sep[lab]};{np.sum(TP[i_best_sep[lab], :, lab])};{np.sum(TN[i_best_sep[lab], :, lab])};{np.sum(FP[i_best_sep[lab], :, lab])};{np.sum(FN[i_best_sep[lab], :, lab])}\n")
    
    def save_treshold_analitics(self, TP, TN, FP, FN, class_combination_dict):
        file_to_save = self.ROC_file
        
        with open(file_to_save, 'a') as f:
            for comb, row_list in class_combination_dict.items():
                f.write(f"{self.file_name};{self.task_name};{self.start_time_str};{self.duration};{self.last_epoch};{comb[0]};{comb[1]};{comb[2]};{np.sum(TP[comb[2], row_list, :])};{np.sum(TN[comb[2], row_list, :])};{np.sum(FP[comb[2], row_list, :])};{np.sum(FN[comb[2], row_list, :])}\n")
                
    def prepare_test_analitics(self, dataset_type = 'validation'):
        file_to_save, sep_file_to_save, sep_tau_file_to_save, dataset = { 
            'test': (self.test_file, self.test_sep_file, self.test_sep_tau_file, self.test_data) 
            , 'atest': (self.atest_file, self.atest_sep_file, self.atest_sep_tau_file, self.atest_data)     
        }[dataset_type]
        prediction = self.model.predict(dataset.batch(8))
        labels = np.zeros(prediction.shape)
        inst_count = prediction.shape[0]
        label_count = prediction.shape[1]
        if type(dataset).__name__ == 'hdf5_file_dataset':            
            for instance, row in enumerate(iter(dataset)):
                labels[instance*dataset.batch_size:(instance+1)*dataset.batch_size, :] = row[1]
        if type(dataset).__name__ ==  '_ZipDataset':       
            for instance, row in enumerate(iter(dataset)):
                labels[instance, :] = row[1].numpy()
            
        TP_separated = np.zeros((label_count))
        TN_separated = np.zeros((label_count))
        FP_separated = np.zeros((label_count))
        FN_separated = np.zeros((label_count))
        
        real_class_list = np.sum(labels, axis = 1)
        class_combination_dict = {}
        
        tau = self.i_best/100.0
        true_positive =  (prediction >= tau) * labels
        true_negative = ((prediction >= tau) - 1) * (labels - 1)
        false_positive = -((prediction >= tau) * (labels - 1))
        false_negative = -((prediction >= tau) - 1) * labels
        predicted_class_list = np.sum((prediction >= tau), axis = 1)
        for inst in range(inst_count):
            try:
                class_combination_dict[(real_class_list[inst], predicted_class_list[inst])].append(inst)
            except KeyError:
                class_combination_dict[(real_class_list[inst], predicted_class_list[inst])] = [inst]      
        
        with open(file_to_save, 'a') as f:
            for comb, row_list in class_combination_dict.items():
                f.write(f"{self.file_name};{self.task_name};{self.start_time_str};{self.duration};{self.last_epoch};{comb[0]};{comb[1]};{self.i_best};{np.sum(true_positive[row_list, :])};{np.sum(true_negative[row_list, :])};{np.sum(false_positive[row_list, :])};{np.sum(false_negative[ row_list, :])}\n")
                
        with open(sep_file_to_save, 'a') as f:
            for task_id in range(label_count):
                for comb, row_list in class_combination_dict.items():
                    f.write(f"{self.file_name};{self.task_name};{self.start_time_str};{self.duration};{self.last_epoch};{comb[0]};{comb[1]};{task_id};{self.i_best};{np.sum(true_positive[row_list, task_id])};{np.sum(true_negative[row_list, task_id])};{np.sum(false_positive[row_list, task_id])};{np.sum(false_negative[ row_list, task_id])}\n")
                    
                    
        true_positive = np.zeros(prediction.shape)
        true_negative = np.zeros(prediction.shape)
        false_positive = np.zeros(prediction.shape)
        false_negative = np.zeros(prediction.shape)
        for task_id in range(label_count):
            tau = self.i_best_sep[task_id]/100.0
            true_positive[:, task_id] =  (prediction[:, task_id] >= tau) * labels[:, task_id]
            true_negative[:, task_id] = ((prediction[:, task_id] >= tau) - 1) * (labels[:, task_id] - 1)
            false_positive[:, task_id] = -((prediction[:, task_id] >= tau) * (labels[:, task_id] - 1))
            false_negative[:, task_id] = -((prediction[:, task_id] >= tau) - 1) * labels[:, task_id]
            if task_id == 0:
                predicted_class_list = np.array((prediction[:, task_id] >= tau), dtype='int')
            else:
                predicted_class_list = predicted_class_list + np.array((prediction[:, task_id] >= tau), dtype='int')
        class_combination_dict = {}
        for inst in range(inst_count):
            try:
                class_combination_dict[(real_class_list[inst], predicted_class_list[inst])].append(inst)
            except KeyError:
                class_combination_dict[(real_class_list[inst], predicted_class_list[inst])] = [inst] 
        
        with open(sep_tau_file_to_save, 'a') as f:
            for task_id in range(label_count):
                for comb, row_list in class_combination_dict.items():
                    f.write(f"{self.file_name};{self.task_name};{self.start_time_str};{self.duration};{self.last_epoch};{comb[0]};{comb[1]};{task_id};{self.i_best_sep[task_id]};{np.sum(true_positive[row_list, task_id])};{np.sum(true_negative[row_list, task_id])};{np.sum(false_positive[row_list, task_id])};{np.sum(false_negative[ row_list, task_id])}\n")  
        
        
        
    def prepare_sep_treshold_analitics(self, dataset_type = 'validation'):
        file_to_save, dataset = { 'validation': (self.ROC_sep_file, self.validation_data)  
        }[dataset_type]
        prediction = self.model.predict(dataset.batch(8))
        labels = np.zeros(prediction.shape)
        inst_count = prediction.shape[0]
        label_count = prediction.shape[1]
        true_positive = np.zeros((101, inst_count, label_count))
        true_negative = np.zeros((101, inst_count, label_count))
        false_positive = np.zeros((101, inst_count, label_count))
        false_negative = np.zeros((101, inst_count, label_count))
        if type(dataset).__name__ == 'hdf5_file_dataset':            
            for instance, row in enumerate(iter(dataset)):
                labels[instance*dataset.batch_size:(instance+1)*dataset.batch_size, :] = row[1]
        if type(dataset).__name__ ==  '_ZipDataset':       
            for instance, row in enumerate(iter(dataset)):
                labels[instance, :] = row[1].numpy()
            
        TP_separated = np.zeros((101, label_count))
        TN_separated = np.zeros((101, label_count))
        FP_separated = np.zeros((101, label_count))
        FN_separated = np.zeros((101, label_count))
        predicted_class_list = np.zeros((101, inst_count))
        
        real_class_list = np.sum(labels, axis = 1)
        class_combination_dict = {}
        
        for i in np.arange(101):
            tau = i/100.0
            true_positive[i, :, :] =  (prediction >= tau) * labels
            true_negative[i, :, :] = ((prediction >= tau) - 1) * (labels - 1)
            false_positive[i, :, :] = -((prediction >= tau) * (labels - 1))
            false_negative[i, :, :] = -((prediction >= tau) - 1) * labels
            predicted_class_list[i, :] = np.sum((prediction >= tau), axis = 1)
            for inst in range(inst_count):
                try:
                    class_combination_dict[(real_class_list[inst], predicted_class_list[i, inst], i)].append(inst)
                except KeyError:
                    class_combination_dict[(real_class_list[inst], predicted_class_list[i, inst], i)] = [inst]      
        
        with open(file_to_save, 'a') as f:
            for comb, row_list in class_combination_dict.items():
                for lab in np.arange(label_count):
                    f.write(f"{self.file_name};{self.task_name};{self.start_time_str};{self.duration};{self.last_epoch};{comb[0]};{comb[1]};{lab};{comb[2]};{np.sum(true_positive[comb[2], row_list, lab])};{np.sum(true_negative[comb[2], row_list, lab])};{np.sum(false_positive[comb[2], row_list, lab])};{np.sum(false_negative[comb[2], row_list, lab])}\n")
                          
        self.save_metrics(true_positive, true_negative, false_positive, false_negative)
        self.save_treshold_analitics(true_positive, true_negative, false_positive, false_negative, class_combination_dict)  
        
    def on_train_end(self, logs=None):
        self.duration = (datetime.now() - self.start_time).total_seconds()
        self.prepare_sep_treshold_analitics('validation')
        if not(self.test_data is None): 
            self.prepare_test_analitics('test')
        if not(self.atest_data is None): 
            self.prepare_test_analitics('atest')
        if self.save_model:
            self.model.save(os.path.join(self.model_save_folder, 'model.h5'), save_format='h5')
            self.model.save(os.path.join(self.model_save_folder, 'model.keras'))
            