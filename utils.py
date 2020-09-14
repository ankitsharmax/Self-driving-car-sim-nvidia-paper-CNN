import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg # RGB operation is performed with matplotlib
from imgaug import augmenters as ia
import cv2
import random

# libraries for model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
lr = 0.0001
loss = 'mse' # the problem is of regression as we are getting continuous data

# get names of files spliiting by '\', basically the last value here
def getName(filePath):
    return filePath.split('\\')[-1]


# importing the data
def importData(path):
    columns = ['Center','Left','Right','Steering','Throttle','Brake','Speed']
    data = pd.read_csv(os.path.join(path,'driving_log.csv'),names = columns)
    #print(data.head())
    #print(data['Center'][0])
    #print(getName(data['Center'][0]))
    data['Center'] = data['Center'].apply(getName)
    #print(data.head())
    print('Image Count: ',data.shape[0])
    #print(len(data['Steering']))
    return data


# data processing
def balanceData(data, display =True):
    nBins = 31
    samplesPerBin = 1000 
    hist, bins = np.histogram(data['Steering'],nBins)
    #print(bins)
    #center = (bins[:-1] + bins[1:])
    #print(center)
    if display:
        center = (bins[:-1] + bins[1:])*0.5 # center the bins to 0
        #print(center)
        plt.bar(center,hist, width = 0.06)
        plt.plot((np.min(data['Steering']),np.max(data['Steering'])),(samplesPerBin,samplesPerBin))
        plt.show()

    # Data cleaning
    removeindexList = []
    for j in range(nBins):
        binDataList = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j + 1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeindexList.extend(binDataList)
 
    print('Removed Images:', len(removeindexList))
    data.drop(data.index[removeindexList], inplace=True)
    print('Remaining Images:', len(data))

    if display:
        hist, _ = np.histogram(data['Steering'], (nBins))
        plt.bar(center, hist, width=0.06)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        plt.show()

    return data


# loading data and coverting to numpy array after some processing
def loadData(path,data):
    imagesPath = []
    steerings = []
    for i in range(len(data)):
        indexedData = data.iloc[i]  # Grabbing data by index
        #print(indexedData)
        imagesPath.append(os.path.join(path,'IMG',indexedData[0]))
        #print(os.path.join(path,'IMG',indexedData[0]))
        steerings.append(float(indexedData[3]))
    imagesPath = np.asarray(imagesPath)
    steerings = np.asarray(steerings)

    return imagesPath, steerings


# image augmentation
def augmentImages(imgPath, steering):
    img = mpimg.imread(imgPath)
    #print(np.random.rand(),np.random.rand(),np.random.rand(),np.random.rand())
    # PAN
    if np.random.rand() < 0.5:
        pan = ia.Affine(translate_percent={'x':(-0.1,0.1),'y':(-0.1,0.1)})
        img = pan.augment_image(img)

    # ZOOM
    if np.random.rand() < 0.5:
        zoom = ia.Affine(scale=(1,1.2))
        img = zoom.augment_image(img)

    # BRIGHTNESS                0 means dark, 1 means normal and above 1 bright
    if np.random.rand() < 0.5:
        brightness = ia.Multiply((0.4,1.2))
        img = brightness.augment_image(img)

    # FLIP
    if np.random.rand() < 0.5:
        img =cv2.flip(img,1)
        steering = -steering
    return img, steering


#imtest, st = augmentImage('test.jpg',-4.5)
#plt.imshow(imtest)
#plt.title(st)
#plt.show()

''' ## use for comparing
def preProcessing(img):
    # roi image
    img = img[60:135,:,:] #H,W
    roiImg = img
    # changing color space to YUV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    return img, roiImg
    
    
imtest, roiImg = preProcessing(mpimg.imread('test.jpg'))
plt.figure(1)
plt.subplot(211)
plt.imshow(imtest)

plt.subplot(212)
plt.imshow(roiImg)
plt.show()
'''

def preProcessing(img):
    # roi image
    img = img[60:135,:,:] #H,W
    #roiImg = img
    # changing color space to YUV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # Gaussian blur         (3,3)- kernel sigmaX = 0
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.resize(img,(200,66))
    img = img/255 # normalization = ranging from 0-1
    return img

#imtest = preProcessing(mpimg.imread('test.jpg'))
#plt.imshow(imtest)
#plt.show()

def batchGen(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []

        for i in range(batchSize):
            index = random.randint(0,len(imagesPath)-1)
            if trainFlag:
                img, steering = augmentImages(imagesPath[index],
                                             steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]
                
            img = preProcessing(img)    
            imgBatch.append(img)
            steeringBatch.append(steering)

        yield(np.asarray(imgBatch), np.asarray(steeringBatch))


def createModel():
    model = Sequential()
    model.add(Convolution2D(24,(5,5),(2,2),input_shape=(66,200,3),activation = 'elu'))
    model.add(Convolution2D(36,(5,5),(2,2),activation = 'elu'))
    model.add(Convolution2D(48,(5,5),(2,2),activation = 'elu'))
    model.add(Convolution2D(64,(3,3),activation = 'elu'))
    model.add(Convolution2D(64,(3,3),activation = 'elu'))

    model.add(Flatten())
    model.add(Dense(100, activation = 'elu'))
    model.add(Dense(50, activation = 'elu'))
    model.add(Dense(10, activation = 'elu'))
    model.add(Dense(1))

    model.compile(Adam(lr),loss)
    return model
    
