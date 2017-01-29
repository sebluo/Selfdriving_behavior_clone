# Load images
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

import cv2
import os
import matplotlib.image as mpimg
#%matplotlib inline

                
    
# load the whole csv as array ,pop up the first line of labels
import csv
train_data_directory = './data/data/IMG/'
val_data_directory='./data/valdata/IMG/'
log_directory='./data/data/'
val_direcory='./data/valdata/'
log_path='./data/data/driving_log.csv'
val_log_path='./data/valdata/driving_log.csv'


logs = []
zero_logs=[]
none_zero_logs=[]
with open(log_path,'rt') as f:
    reader = csv.reader(f)
    for line_data in reader:
        logs.append(line_data)

        
log_labels = logs.pop(0)
for i in range(len(logs)):
    if float(logs[i][3])==0.0:
        zero_logs.append(logs[i])
    else:
        none_zero_logs.append(logs[i])

def log_select(zero_logs,none_zero_logs,zero_factor=1,train_factor=0.9):
    random.shuffle(zero_logs)
    random.shuffle(none_zero_logs)
    
    num_train_zero_log=int(zero_factor*train_factor*len(zero_logs))
    num_train_non_zero_log=int(train_factor*len(none_zero_logs))
    num_train_log=num_train_zero_log+num_train_non_zero_log

    num_val_zero_log=int(zero_factor*(1-train_factor)*len(zero_logs))
    num_val_non_zero_log=len(none_zero_logs)-num_train_non_zero_log-1
    num_val_log=num_val_zero_log+num_val_non_zero_log

    train_logs=zero_logs[0:num_train_zero_log]+none_zero_logs[0:num_train_non_zero_log]
    val_part1=random.sample(zero_logs[num_train_zero_log:-1],num_val_zero_log)
    val_part2=random.sample(none_zero_logs[num_train_non_zero_log:-1],num_val_non_zero_log)
    val_logs=val_part1+val_part2

    #print(num_train_zero_log)
    #print(num_train_non_zero_log)
    #print(num_val_zero_log)
    #print(num_val_non_zero_log)
    random.shuffle(train_logs)
    random.shuffle(val_logs)
    return train_logs, val_logs




def get_steering_from_logs(logs):
    steering_list=[]
    for i in range(len(logs)):
        steering_angle=float(logs[i][3].strip())
        steering_list.append(steering_angle)
    return steering_list

#pre processing
import math
col,row=320,160
#new_size_col,new_size_row = 64, 32
new_size_col,new_size_row = 200, 66
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    #image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def trans_image(image,steer,trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*0.4
    tr_y = 40*np.random.uniform()-40/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(col,row))
    return image_tr,steer_ang

def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image


def preprocessImage(image):
    shape = image.shape
    # note: numpy arrays are (row, col)!
    image = image[math.floor(shape[0]/5):shape[0]-20, 0:shape[1]]
    image = cv2.resize(image,(new_size_col,new_size_row),interpolation=cv2.INTER_AREA)    
    return image

def preprocess_image_random(line_data):
    i_lrc = np.random.randint(3)
    if (i_lrc <= 0):
        path_file = log_directory+line_data[0].strip()
        shift_ang = 0
    if (i_lrc == 1):
        path_file = log_directory+line_data[1].strip()
        shift_ang = 0.25
    if (i_lrc == 2):
        path_file = log_directory+line_data[2].strip()
        shift_ang = -.25
    y_steer = float(line_data[3].strip()) + shift_ang
    #image = cv2.imread(path_file)
    #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = plt.imread(path_file)
    image,y_steer = trans_image(image,y_steer,50)
    image = augment_brightness_camera_images(image)
    image = preprocessImage(image)
    #image=add_random_shadow(image)
    image = np.array(image)
    
    ind_flip = np.random.randint(2)
    if ind_flip==0:
        image = cv2.flip(image,1)
        y_steer = -y_steer
    #image = image.astype(np.float32)
    #image = image/255.0 - 0.5
    return image,y_steer

def generateBatch(logs,batch_size = 32,threshhold=0.5):
    rounder_factor=1
    batch_images = np.zeros((batch_size, new_size_row, new_size_col, 3))
    batch_steering = np.zeros(batch_size)
    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(len(logs))
            line_data = logs[i_line]
            x,y = preprocess_image_random(line_data)
            """
            keep_pr = 0
            while keep_pr == 0:
            	i_line = np.random.randint(len(logs))
            	line_data = logs[i_line]
            	x,y = preprocess_image_random(line_data)
                
            	if abs(y)<0.01:
                    pr_val = np.random.uniform()
                    if pr_val>threshhold:
                        keep_pr = 1

            	if abs(y)>0.1:
                	y=y*rounder_factor
                	keep_pr = 1
            	else:
            		keep_pr=1
            
                """
            if abs(y)>0.4 and (i_batch+9)<batch_size:
                for j in range(10):
                    batch_images[i_batch] = x
                    batch_steering[i_batch] = y
                    i_batch=i_batch+1
            else:        
                batch_images[i_batch] = x
                batch_steering[i_batch] = y
        yield batch_images, batch_steering


def generateBatchVal(val_logs,batch_size = 32):
    batch_images = np.zeros((batch_size, new_size_row, new_size_col, 3))
    batch_steering = np.zeros(batch_size)
    
    startIdx = 0
    batchCount = len(val_logs)/batch_size 
    while True: 
        endIdx = startIdx + batch_size

        for i in range(batch_size):
            #img_path=val_data_directory+val_logs[startIdx+i][0].split("D:\\selfdriving\\behavior cloning video and project\\simulator-windows-64\\IMG\\")[1].strip()
            #i_line = np.random.randint(len(val_logs))
            i_line=startIdx+i
            line_data = val_logs[i_line]
            path_file = log_directory+line_data[0].strip()
            img = plt.imread(path_file)
            x=preprocessImage(img)
            y=float(line_data[3])
            batch_images[i] = x
            batch_steering[i] = y
        yield batch_images, batch_steering
        startIdx = endIdx
        if len(val_logs)-startIdx <batch_size:
            startIdx = len(val_logs)-batch_size-1


# visualize the generateBatch steering distribution
def get_steeringList(batch_generator,size):
    steeringList = np.empty((0))
    for i in range(1,size):
        steeringList = np.append(steeringList,next(batch_generator)[1])
        print(next(batch_generator)[1])
    return steeringList


# model design
import pickle
import numpy as np
import math
from keras.utils import np_utils
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout, Dense, Activation
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.regularizers import l2

model = Sequential()


def build_model1(model):    #input shape (33,100,3)
    reg_val=0.0001
    inputShape = (66,200,3)
     # select elu because output can be less than zero 
    model.add(Lambda(lambda x: x/255.-0.5,input_shape=inputShape,output_shape=inputShape,name="Normalization"))
    model.add(Conv2D(24,5,5,subsample=(2,2), activation = 'elu', border_mode='valid',W_regularizer=l2(reg_val),name='Conv2D1')) 
    model.add(Dropout(0.5, name='DropoutC1'))
    #output shape 31*98*24*3

    model.add(Conv2D(36,5,5,subsample=(2,2), activation = 'elu',  border_mode='valid',name='Conv2D2'))
    model.add(Dropout(0.5, name='DropoutC2'))
    #output shape 14*47*36*3

    model.add(Conv2D(48,5,5,subsample=(2,2), activation = 'elu', border_mode='valid', name='Conv2D3'))
    model.add(Dropout(0.5, name='DropoutC3'))
    #output shape 5*22*48*3

    model.add(Conv2D(64,3,3,subsample=(1,1), activation = 'elu', border_mode='valid', name='Conv2D4'))
    model.add(Dropout(0.5, name='DropoutC4'))
    #output shape 3*20*64*3

    model.add(Conv2D(64,3,3,subsample=(1,1), activation = 'elu', border_mode='valid', name='Conv2D5'))
    model.add(Dropout(0.5, name='DropoutC5'))
    #output shape 1*18*64*3

    # convolution to dense,flatten layer  1152 nodes
    model.add(Flatten(name='flatten'))

    model.add(Dense(100,activation='elu', name='Dense1'))
    model.add(Dropout(0.5, name='DropoutD1'))

    model.add(Dense(50,activation='elu', name='Dense2'))
    model.add(Dropout(0.5, name='DropoutD2'))

    model.add(Dense(10,activation='elu', name='Dense3'))
    model.add(Dropout(0.5, name='DropoutD3'))

    model.add(Dense(1,activation='elu', name='Output')) # problem is a regression

    model.summary()
    
build_model1(model)



def predict_test(i,steering_angle):
        
        path_file = log_directory+logs[i][0].strip()
        image = plt.imread(path_file)
        image_array = np.asarray(image)
        image_array=preprocessImage(image_array)
        transformed_image_array = image_array[None, :, :, :]
        predicted_steering_angle = float(model.predict(transformed_image_array, batch_size=1))
        print(" logs in row ",i," steering_angle is ",steering_angle," predicted_steering_angle is ",predicted_steering_angle)

#adam = Adam(lr=1e-4)
#model.compile(loss='mse',optimizer=adam)
model.compile(optimizer="adam", loss="mse")

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

import os
if os.path.exists("./model.h5"):
    model.load_weights("./model.h5")
    print("load the previous model.h5 successfully")


from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback, Callback
import json

numTimes = 8
numEpoch = 1


#train_logs, val_logs=log_select(zero_logs,none_zero_logs,zero_factor=1)
#model.load_weights('./throttle 0.1.h5')
for time in range(numTimes):
    #train_logs, val_logs=log_select(zero_logs,none_zero_logs,zero_factor=1)
    trainGenerator = generateBatch(logs,250)
    validGenerator = generateBatchVal(logs,250)
    samplesPerEpoch = 20000 
    nbValSamples = 8250
    #history = model.fit_generator(trainGenerator, samplesPerEpoch, numEpoch, verbose=1)
    #history = model.fit_generator(trainGenerator, samplesPerEpoch, numEpoch,verbose=1, validation_data=validGenerator, nb_val_samples = nbValSamples,
    #                callbacks=[ModelCheckpoint(filepath="bestVal.h5", verbose=1, save_best_only=True), ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=0.000001)])
    history = model.fit_generator(trainGenerator, samples_per_epoch=samplesPerEpoch, nb_epoch=numEpoch, validation_data=validGenerator,
                   nb_val_samples=nbValSamples, callbacks=[ModelCheckpoint(filepath="modelweights"+str(time)+".h5", verbose=1, save_best_only=True)])
                
    #history = model.fit_generator(trainGenerator, samples_per_epoch=samplesPerEpoch, nb_epoch=numEpoch,  callbacks=[ModelCheckpoint(filepath="bestVal"+str(time)+".h5", verbose=1, save_best_only=True)])
                

    print("the epoch",time)

    for k in range(30):
        steering_angle=0

        while(abs(steering_angle)<0.5):
            i = np.random.randint(len(logs))
            steering_angle=float(logs[i][3].strip())
        predict_test(i,steering_angle)    
    #threshhold=threshhold/(time+1)


    
        
    

    
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Model saved.")
    