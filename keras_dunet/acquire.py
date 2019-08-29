import cv2
import matplotlib.pyplot as plt
import numpy as np
from agumentation import flip1

def get_x(m,num):
    rows, cols = m.shape
    x1 = 0
    x2 = 0
    change = 0
    for i in range(cols):
        count = 0
        for j in range(rows):
            if m[j][i] == num:
                count = count+1
        if change ==0:
            if count!=0:
                x1=i-1
                change = 1
        else:
            if count==0:
                x2 = i
                if (x2-x1)<28:
                    change = 0
                    continue
                break
    return x1,x2

def get_y(m,num,x1,x2):
    rows, cols = m.shape
    sa = abs(x1-x2)
    y1 = 0
    y2 = 0
    change = 0
    for i in range(rows):
        count = 0
        for j in range(sa):
            if m[i][j+x1] == num:
                count = count+1
        if change ==0:
            if count!=0:
                y1=i-1
                change = 1
        else:
            if count==0:
                y2 = i
                if (y2-y1)<28:
                    change = 0
                    continue
                break
    return y1,y2

def get_location_mask(img,location):
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            if img[i][j] == location:
                img[i][j] = location
            else:
                img[i][j] = 0
    return img.astype(np.uint8)

def output(mask,original,location):
    for i in range(224):
        for j in range(224):
            if mask[i][j] != location:
                original[i][j][0] = 0
                original[i][j][1] = 0
                original[i][j][2] = 0
    return original

def get_hehe(img,location,name,root):
    img = get_location_mask(img, location)
    # img = flip1(img)
    x1,x2=get_x(img,location)
    print(name,x1,x2)
    y1,y2= get_y(img,location,x1,x2)
    print(y1,y2)
    img = img[y1:y2,x1:x2]
    if (x1-x2)*(y1-y2)==0:
        print("hehe")
    else:
        mask = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        pic = cv2.imread(root+name)
        # pic = flip1(pic)
        rows, cols, p = pic.shape
        new_x1 = int(cols*x1/256)
        new_x2 = int(cols*x2/256)
        new_y1 = int(rows*y1/256)
        new_y2 = int(rows*y2/256)
        pic = pic[new_y1:new_y2,new_x1:new_x2]
        pic = cv2.resize(pic, (224, 224), interpolation=cv2.INTER_LINEAR)
        final = output(mask,pic,location)
        # final = flip1(final)
        cv2.imwrite("demo/outline/"+name,final)