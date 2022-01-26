import os
import sys
import glob
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, AvgPool1D, GlobalAvgPool1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM, Dense,Dropout, Bidirectional
from tensorflow.keras.layers import TimeDistributed
import keras_tuner as kt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
n_features = 12
length = 5000
out_dims = 2
pooling1 = 2

# data func
filepaths_xml = sorted(glob.glob('/data/Farah/0_data/ECG_LVH/01_dataset/zxz_f/*.XML'))
for i in range(len(filepaths_xml)):
        xml_filepath = str(filepaths_xml[i])
        # parse xml
        tree = ET.parse(xml_filepath)
        root = tree.getroot()
        leads_12 = []
        for elem in tree.iterfind('./%scomponent/%sseries/%scomponent/%ssequenceSet/%scomponent/%ssequence'%(tag_pre,tag_pre,tag_pre,tag_pre,tag_pre,tag_pre)):
            for child_of_elem in elem:
                if child_of_elem.tag == '%scode'%(tag_pre):
                    for grand_child_digits in elem.iterfind('%svalue/%sdigits'%(tag_pre,tag_pre)):
                        arr = grand_child_digits.text.split(' ')
                        num_samples = np.array(arr).shape[0]
                        leads_12.append(arr)                        
        leads_12 = np.array(leads_12) # (12, 31600)
        leads_12_dim = leads_12.transpose(1,0) # (31600, 12)
        
        for idx in range(0, leads_12.shape[-1] - cut_size, step_pix):
            sample = leads_12_dim[idx:idx + cut_size, :]
            sample = sample.astype(np.float32)
            mean = np.mean(sample)
            std = np.std(sample)
            if std > 0:
                ret = (sample - mean) / std
            else:
                ret = sample * 0
            all_leads.append(ret)  
        all_leads = np.array(all_leads)
        np.save(filepath_npy, all_leads)
    
# train
for in_dim in [1,2,5,10, 20, 25,50, 100, 200, 250, 500]: 
    
    len_frame = int(length / in_dim)

    for test_index in [0,1,2,3,4]: 
        
        val_index = test_index-1 if test_index-1>=0 else 4

        x_train = np.load(filepath_npy)
        y_train = np.load(filepath_npy.replace('train_data', 'train_label'))

        x_val = np.load(filepath_npy)
        y_val = np.load(filepath_npy.replace('val_data', 'val_label'))

        x_test = np.load(filepath_npy)
        y_test = np.load(filepath_npy.replace('test_data', 'test_label'))

        x_train = x_train.reshape((-1, in_dim, len_frame, n_features))
        x_val = x_val.reshape((-1, in_dim, len_frame, n_features))
        x_test = x_test.reshape((-1, in_dim, len_frame, n_features))

        tuner = kt.Hyperband(
            build_model,
            objective='val_accuracy',
            max_epochs=30,
            hyperband_iterations=1)

        tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val), 
                 callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)])

        best_hps = tuner.get_best_hyperparameters(1)
        model = build_model(best_hps[0])

        test_tuner(model, x_test, y_test, dirpath_project, dirpath_report, dirpath_matrix, title_auc, dirpath_auc, str_param, test_index, val_index, class_names)

# matrics func
report = classification_report(y_true_test, y_pred_test, target_names=class_names)
cm = confusion_matrix(y_true, y_pred)
fpr, tpr, thresholds1 = metrics.roc_curve(y_test_onehot.ravel(), y_pred_pro.ravel())
auc = metrics.auc(fpr, tpr)
