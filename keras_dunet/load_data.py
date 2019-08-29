from random import shuffle
import os
import cv2
import numpy as np
import queue
from keras.utils import to_categorical
from agumentation import *

train = queue.Queue()
test = queue.Queue()
root1 = "data/lung_train/"
root2 = "data/lung_label/"
root3 = "data/test_lung/"
root4 = "data/test_label/"

def get_name(root):
    for root, dirs, SB in os.walk(root):
        return SB

def load_name():
    print("load train")
    name_train = get_name(root1)
    shuffle(name_train)
    for i in range(len(name_train)):
        train.put(name_train[i])
        if i == len(name_train)-1:
            print(train.qsize())

    print("load test")
    name_test = get_name(root3)
    shuffle(name_test)
    for i in range(len(name_test)):
        test.put(name_test[i])
        if i == len(name_test)-1:
            print(test.qsize())

def get_batch_train(batch_size):
    batch_train = []
    batch_label = []
    for i in range(batch_size):
        name = train.get()
        train.put(name)
        pic_train = cv2.imread(root1+name,cv2.IMREAD_GRAYSCALE)  #get image
        pic_train = cv2.resize(pic_train, (256, 256), interpolation=cv2.INTER_LINEAR)

        name_label = name[0:-3] #get label
        pic_label = cv2.imread(root2+name_label+"png",cv2.IMREAD_GRAYSCALE)
        pic_label = cv2.resize(pic_label, (256, 256), interpolation=cv2.INTER_LINEAR)

        pic_train,pic_label = aug(pic_train,pic_label) #agumentation
        batch_train.append([pic_train / 255])
        batch_label.append([pic_label])
    return np.reshape(np.concatenate(batch_train),(-1,256,256,1)),to_categorical(np.reshape(np.concatenate(batch_label),(-1,256,256)))

def get_batch_test(batch_size):
    batch_test = []
    batch_label = []
    for i in range(batch_size):
        name = test.get()
        test.put(name)
        pic_train = cv2.imread(root3+name,cv2.IMREAD_GRAYSCALE)
        pic_train = cv2.resize(pic_train, (256, 256), interpolation=cv2.INTER_LINEAR)
        batch_test.append([pic_train/255])

        name_label = name[0:-3]
        pic_label = cv2.imread(root4+name_label+"png",cv2.IMREAD_GRAYSCALE)
        pic_label = cv2.resize(pic_label, (256, 256), interpolation=cv2.INTER_LINEAR)
        batch_label.append([pic_label])
    return np.reshape(np.concatenate(batch_test),(-1,256,256,1)),to_categorical(np.reshape(np.concatenate(batch_label),(-1,256,256)))

def get_demo():
    pic = cv2.imread("0.jpg",cv2.IMREAD_GRAYSCALE)
    pic = cv2.resize(pic, (256, 256), interpolation=cv2.INTER_LINEAR)
    pic = pic/255
    map = cv2.imread("0.jpg")
    map = cv2.resize(map, (256, 256), interpolation=cv2.INTER_LINEAR)
    return np.reshape(pic,(-1,256,256,1)),map

