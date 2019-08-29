import cv2
import numpy as np

def rd():
    a = np.random.randint(-25,26)
    b = np.random.randint(-25,26)
    return a,b

def tranlate(train,label):
    a,b=rd()
    rows, cols = train.shape
    M = np.float32([[1, 0, a], [0, 1, b]])
    dst1 = cv2.warpAffine(train, M, (cols, rows))
    dst2 = cv2.warpAffine(label, M, (cols, rows))
    return dst1,dst2

def flip1(img):
    image = cv2.flip(img,1)
    return image

def flip(train,label):
    train = cv2.flip(train,1)
    label = cv2.flip(label,1)
    return train,label

def aug(train,label):
    a = np.random.randint(0, 2)
    if a ==0:
        train,label = flip(train,label)
    train, label = tranlate(train,label)

    return train,label