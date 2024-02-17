
"""
Feature extract with pretrained CNN
"""

import os
import numpy as np
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import DenseNet201

print("tf version {}".format(tf.__version__))
if tf.test.is_gpu_available():
    print(tf.test.gpu_device_name())
else:
    print("TF cannot find GPU")
    
base_dir = 'DataSet'
train_dir = os.path.join(base_dir,'Train')
test_dir = os.path.join(base_dir,'Test')

datagen = ImageDataGenerator(rescale = 1./255)
batch_size = 16

conv_base = DenseNet201(weights = 'imagenet',      
                 include_top = False,        
                 input_shape = (224,224,3))
    

def extract_features(directory, sample_size):
    
    features = np.zeros(shape = (sample_size,7,7,1920))  
    labels = np.zeros(shape = (sample_size, 4))          
    
    generator = datagen.flow_from_directory(
        directory,
        target_size = (224,224),
        batch_size = batch_size,
        class_mode = 'categorical')
    
    i = 0
    for inputs_batch, labels_batch in generator:
        
        features_batch = conv_base.predict(inputs_batch)
        features[i*batch_size : (i+1)*batch_size] = features_batch
        labels[i*batch_size : (i+1)*batch_size] = labels_batch
        i += 1
        
        if i*batch_size >= sample_size:
            break
        
    return features, labels


train_features, train_labels = extract_features(train_dir, 537)
test_features, test_labels = extract_features(test_dir, 60)

X_train = np.reshape(train_features, (537, 7*7*1920))
y_train = train_labels

X_test = np.reshape(test_features, (60, 7*7*1920))
y_test = test_labels

np.save('tr_f_Dense201',X_train)
np.save('tr_l_Dense201',y_train)
np.save('tst_f_Dense201',X_test)
np.save('tst_l_Dense201',y_test)