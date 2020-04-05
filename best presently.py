import os,sys
import numpy as np
import scipy
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

from PIL import Image
import random
import keras
from keras.utils import np_utils


from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

def DataSet():
    
    train_path_glue ='C:/新建文件夹/keras trainning project 1/dataset/train/tank/'
    train_path_medicine = 'C:/新建文件夹/keras trainning project 1/dataset/train/none/'
    
    test_path_glue ='C:/新建文件夹/keras trainning project 1/dataset/test/tank/'
    test_path_medicine = 'C:/新建文件夹/keras trainning project 1/dataset/test/none/'
    
    imglist_train_glue = os.listdir(train_path_glue)
    imglist_train_medicine = os.listdir(train_path_medicine)
    
    imglist_test_glue = os.listdir(test_path_glue)
    imglist_test_medicine = os.listdir(test_path_medicine)
        
    X_train = np.empty((len(imglist_train_glue) + len(imglist_train_medicine), 224, 224, 3))
    Y_train = np.empty((len(imglist_train_glue) + len(imglist_train_medicine), 2))
    count = 0
    for img_name in imglist_train_glue:
        
        img_path = train_path_glue + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count] = img
        Y_train[count] = np.array((1,0))
        count+=1
        
    for img_name in imglist_train_medicine:

        img_path = train_path_medicine + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count] = img
        Y_train[count] = np.array((0,1))
        count+=1
        
    X_test = np.empty((len(imglist_test_glue) + len(imglist_test_medicine), 224, 224, 3))
    Y_test = np.empty((len(imglist_test_glue) + len(imglist_test_medicine), 2))
    count = 0
    for img_name in imglist_test_glue:

        img_path = test_path_glue + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test[count] = np.array((1,0))
        count+=1
        
    for img_name in imglist_test_medicine:
        
        img_path = test_path_medicine + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test[count] = np.array((0,1))
        count+=1
        
    index = [i for i in range(len(X_train))]
    random.shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]
    
    index = [i for i in range(len(X_test))]
    random.shuffle(index)
    X_test = X_test[index]    
    Y_test = Y_test[index]

    return X_train,Y_train,X_test,Y_test



X_train,Y_train,X_test,Y_test = DataSet()
print('X_train shape : ',X_train.shape)
print('Y_train shape : ',Y_train.shape)
print('X_test shape : ',X_test.shape)
print('Y_test shape : ',Y_test.shape)
'''
X_train = X_train.reshape(-1, 3,224, 224)/255.
X_test = X_test.reshape(-1, 3,224, 224)/255.
Y_train = np_utils.to_categorical(Y_train, num_classes=2)
Y_test = np_utils.to_categorical(Y_test, num_classes=2)
'''
model = Sequential()

rate1=0.35
rate2=0.25
# Conv layer 1 output shape (32, 224, 224)

model.add(Conv2D(filters=32,kernel_size=(20,20),
                 input_shape=(224,224,3), 
                 activation='relu', 
                 padding='same'))
model.add(Dropout(rate1))
model.add(MaxPooling2D(pool_size=(16, 16))) # 16* 16y_test

model.add(Conv2D(filters=16, kernel_size=(10, 10), 
                 activation='relu', padding='same'))
model.add(Dropout(rate1))
model.add(MaxPooling2D(pool_size=(8, 8))) # 8 * 8

model.add(Flatten()) # FC1,64个8*8转化为1维向量
for term in range(1):
    model.add(Dropout(rate2))
    model.add(Dense(256, activation='relu')) # FC2 1024
model.add(Dropout(rate2))
model.add(Dense(2, activation='softmax')) # Output 10

model.summary()

opt = keras.optimizers.rmsprop(lr=0.0001,decay=1e-6)

model.compile(loss='categorical_crossentropy',
             optimizer=opt,
             metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range = 20,
    zoom_range = 0.15,
    horizontal_flip = True,
)

model.fit_generator(datagen.flow(X_train,Y_train,batch_size=16),steps_per_epoch = 100,epochs = 5,validation_data=(X_test,Y_test),workers=4,verbose=1)
model=tf.keras.models.load_model('train1_trained_model.h5')

scores = model.evaluate(X_test,Y_test,verbose=1)
print('Test loss:',scores[0])
print('Test accuracy:',scores[1])
