import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_lr(optimizer):
    def lr(y_true,y_pred):
        return optimizer.lr
    return lr

def LR(epoch):
    if epoch == 10:
        return True
    if epoch == 15:
        return True



def back(m,rols,cols):
    a =  np.zeros([rols,cols,3])
    for i in range(rols):
         for o in range(cols):
            if m[0][i][o] == 1:
                a[i][o][0] = 128
                a[i][o][1] = 128
                a[i][o][2] = 0
            elif m[0][i][o] == 2:
                a[i][o][0] = 0
                a[i][o][1] = 0
                a[i][o][2] = 128
            elif m[0][i][o] == 3:
                a[i][o][0] = 0
                a[i][o][1] = 128
                a[i][o][2] = 0
            elif m[0][i][o] == 4:
                a[i][o][0] = 128
                a[i][o][1] = 0
                a[i][o][2] = 0
            else:
                a[i][o][0] = 0
                a[i][o][1] = 0
                a[i][o][2] = 0
    return a.astype(np.uint8)

def show(origin,label):
    image = cv2.addWeighted(origin, 1.0, label, 0.6, 0)
    image = image[:, :, (2, 1, 0)]
    cv2.imwrite('demo/11.jpg' , image)
    plt.figure("Image",facecolor='#FFFFFF')
    plt.imshow(image)
    plt.axis('on')
    plt.title("hehe")
    plt.show()

def show_demo(demo_in,demo_origin):
    label = back(demo_in,256,256)
    show(demo_origin,label)


def iou(predict,true,batch):
    iou = []
    for q in range(4):
        count = 0
        for o in range(batch):
            p = 0
            t = 0
            mix = 0
            for i in range(256):
                for j in range(256):
                    if predict[o][i][j]== q+1:
                        p = p+1
                    if true[o][i][j]== q+1:
                        t = t+1
                    if (predict[o][i][j]== q+1)&(true[o][i][j]== q+1):
                        mix = mix+1
            gg = mix/(p+t-mix)
            count = count+gg
        iou.append(count/batch)
    return iou