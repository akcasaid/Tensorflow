# -*- coding: utf-8 -*-
"""transfer-learning-k-fold-oct.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WIK7Z4w7ibn-kCtodMH0K-StaaxoHoJc
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet201, Xception, InceptionResNetV2, VGG19, VGG16, MobileNet, ResNet152V2
#from keras_efficientnets import EfficientNetB2
from tensorflow.keras.models import Model
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras import optimizers
from tensorflow.keras import  callbacks

base_dir = 'data_path'
ca_dir = os.path.join(base_dir,'class1')
ch_dir = os.path.join(base_dir,'class2')
pet_dir = os.path.join(base_dir,'class2')
son_dir=os.path.join(base_dir,"class3")

ca_fnames = [os.path.join(ca_dir, fname) for fname in os.listdir(ca_dir)]
ch_fnames = [os.path.join(ch_dir, fname) for fname in os.listdir(ch_dir)]
pet_fnames = [os.path.join(pet_dir, fname) for fname in os.listdir(pet_dir)]
son_fnames = [os.path.join(son_dir, fname) for fname in os.listdir(son_dir)]


# img = plt.imread(ca_fnames[2])
# plt.imshow(img)
# plt.show()

def creat_dataframe(fnames):

    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], fnames))

    fnames = pd.Series(fnames, name = 'filePath').astype(str)
    labels = pd.Series(labels, name = 'label')

    df = pd.concat([fnames, labels], axis = 1)

    df = df.sample(frac = 1).reset_index(drop= True)

    return df

ca_df = creat_dataframe(ca_fnames)
ch_df = creat_dataframe(ch_fnames)
pet_df = creat_dataframe(pet_fnames)
son_df= creat_dataframe(son_fnames)

df = pd.concat([ca_df, ch_df, pet_df, son_df], axis = 0)

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.20,
    zoom_range = 0.20,
    horizontal_flip = True,
    fill_mode = 'nearest')

val_datagen = ImageDataGenerator(rescale = 1./255)

"""model_checkpoint_dir= 'model_checkpoint'
try:
    # Create target Directory
    os.mkdir(model_checkpoint_dir)
    print("Directory " , model_checkpoint_dir ,  " Created ")
except FileExistsError:
    print("Directory " , model_checkpoint_dir ,  " already exists")


csv_dir= 'csv'
try:
    # Create target Directory
    os.mkdir(csv_dir)
    print("Directory " , csv_dir ,  " Created ")
except FileExistsError:
    print("Directory " , csv_dir ,  " already exists")

LCurve_dir= 'LCurve'
try:
    # Create target Directory
    os.mkdir(LCurve_dir)
    print("Directory " , LCurve_dir ,  " Created ")
except FileExistsError:
    print("Directory " , LCurve_dir ,  " already exists")
    """

def creat_model():

    conv_base = ResNet152V2(weights = 'imagenet',
                        include_top = False,
                        input_shape = (150,150,3))


    x = conv_base.output
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x_output = layers.Dense(4, kernel_regularizer=tf.keras.regularizers.l2(0.01),activation
             ='softmax')(x)

    model = Model(inputs = conv_base.input, outputs = x_output)

    model.compile(loss= 'squared_hinge',
              optimizer = optimizers.Adam(learning_rate=2e-5),
              metrics = ['acc'])

    #RMSprop(lr = 2e-5)
    return model

kfold = StratifiedShuffleSplit(n_splits=5, test_size=0.177, random_state=0)

inputs = df.filePath.values
targets = df.label.values
model_history = []

# K-fold Cross Validation
fold_no = 1
for train, test in kfold.split(inputs, targets):

  training_data = df.iloc[train]
  validation_data = df.iloc[test]

  train_data_generator = train_datagen.flow_from_dataframe(training_data,
                               x_col = "filePath", y_col = "label", target_size = (150,150),
                               batch_size = 8,
                               class_mode = "categorical", shuffle = True)
  val_data_generator  = val_datagen.flow_from_dataframe(validation_data,
                            x_col = "filePath", y_col = "label", target_size = (150,150),
                            batch_size = 8,
                            class_mode = "categorical", shuffle = True)


  train_data_generator.reset()
  val_data_generator.reset()
  print('-'*70)
  print(f'Training for fold {fold_no} ...')

  #history = fit_and_evaluate(train_data_generator, val_data_generator)

  keras.backend.clear_session()

  callbacks_list = [
     callbacks.EarlyStopping( monitor = 'acc',
                           patience = 5,),
     callbacks.ModelCheckpoint('./ResNet152V2.h5',
                            monitor='val_acc', verbose=1,
                            save_best_only=True, mode='max'),
     callbacks.ReduceLROnPlateau( monitor='val_loss',
                               factor = 0.1,
                               patience = 5,),
     callbacks.History()
   ]

  model = None
  model = creat_model()
  history = model.fit(train_data_generator,
                validation_data=val_data_generator,
                epochs=100,
                callbacks=callbacks_list)
  print("Val Score: ", model.evaluate(val_data_generator))
  model_history.append(history)

  #print(history.history)
  train_data_generator.reset()
  val_data_generator.reset()

  fold_no = fold_no + 1

"""pd.DataFrame(model_history[0].history).to_csv("csv/ResNet152V2_1.csv", index = False)
pd.DataFrame(model_history[1].history).to_csv("csv/ResNet152V2_2.csv", index = False)
pd.DataFrame(model_history[2].history).to_csv("csv/ResNet152V2_3.csv", index = False)
pd.DataFrame(model_history[3].history).to_csv("csv/ResNet152V2_4.csv", index = False)
pd.DataFrame(model_history[4].history).to_csv("csv/ResNet152V2_5.csv", index = False)  """

fig = plt.figure()
#figsize=(15,5)
epoch0=range(1,len(model_history[0].history['acc'])+1)
epoch1=range(1,len(model_history[1].history['acc'])+1)
epoch2=range(1,len(model_history[2].history['acc'])+1)
epoch3=range(1,len(model_history[3].history['acc'])+1)
epoch4=range(1,len(model_history[4].history['acc'])+1)
plt.title('ResNet152V2')
plt.plot(epoch0,model_history[0].history['acc'], label='Train Fold 1', color='black')
plt.plot(epoch0,model_history[0].history['val_acc'], label='Val Fold 1', color='black', linestyle = "dashdot")
plt.plot(epoch1,model_history[1].history['acc'], label='Train Fold 2', color='red', )
plt.plot(epoch1,model_history[1].history['val_acc'], label='Val Fold 2', color='red', linestyle = "dashdot")
plt.plot(epoch2,model_history[2].history['acc'], label='Train Fold 3', color='green', )
plt.plot(epoch2,model_history[2].history['val_acc'], label='Val Fold 3', color='green', linestyle = "dashdot")
plt.plot(epoch3,model_history[3].history['acc'], label='Train Fold 4', color='blue', )
plt.plot(epoch3,model_history[3].history['val_acc'], label='Val Fold 4', color='blue', linestyle = "dashdot")
plt.plot(epoch4,model_history[4].history['acc'], label='Train Fold 5', color='orange', )
plt.plot(epoch4,model_history[4].history['val_acc'], label='Val Fold 5', color='orange', linestyle = "dashdot")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('./ResNet152V2.png',bbox_inches='tight', dpi = 600)
plt.show()



