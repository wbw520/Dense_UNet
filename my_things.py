import numpy as np
import warnings
import keras
from keras.callbacks import Callback
from keras import backend as K
from sklearn import metrics
import matplotlib.pyplot as plt
import cv2
import math
import config
import itertools
from keras.utils import to_categorical

C = config.Config()

#It is a method for getting learning_rate in every update
def get_lr(optimizer):
    def lr(y_true,y_pred):
        return optimizer.lr
    return lr


#It is a class in order to generate the list of data
#having structure of [[image,label]......]
#it will be used for class generate later
class prepare_list():
    def __init__(self,root):
        self.root = root
        self.mode = C.mode

    def read_txt(self):
        with open(self.root,"r",encoding="UTF-8") as data:
            name =[]
            lines = data.readlines()
            for line in lines:
                a = line.split(",")[0]
                b = line.split(",")[1]
                label = self.deal_label(b[:-1])
                name.append([a,label])
            return name

    def read_txt2(self):
        with open(self.root,"r",encoding="UTF-8") as data:
            name =[]
            lines = data.readlines()
            for line in lines:
                a = line[:-1]
                name.append(a)
            return name

    def deal_label(self,data):
        label = 0
        if self.mode == "right":
            if int(data[0])+int(data[2])+int(data[4])+int(data[6]) >0:
                label = 1
        if self.mode == "left":
            if int(data[1])+int(data[3])+int(data[5])+int(data[7]) >0:
                label = 1
        if self.mode == "total":
            if data != "00000000":
                label = 1
        return label

#generate data for keras training,seperate model
class DataGenerator_sp(keras.utils.Sequence):
    def __init__(self, datas, batch_size=C.batch_size,shuffle=True):
        self.batch_size = batch_size
        self.datas = datas
        self.indexes = np.arange(len(self.datas))
        self.shuffle = shuffle
        self.mode = C.mode

    #this will give the iteration numbers for one epoch
    def __len__(self):
        return math.ceil(len(self.datas) / float(self.batch_size))-1

    def __getitem__(self, item):
        # generate one batch with item
        batch_indexs = self.indexes[item * self.batch_size:(item + 1) * self.batch_size]

        # get data from datas collection by batch_indexs
        batch_datas = [self.datas[k] for k in batch_indexs]
        x, y = self.data_generation(batch_datas)
        return x,y

    def on_epoch_end(self):
        # whether use random for indexes after one epoch
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_datas):
        images = []
        labels = []

        for i, data in enumerate(batch_datas):
            # generate image data
            image= self.deal(data[0])
            images.append(image/255)
            # generate label
            label = data[1]
            labels.append(label)

        return np.array(images, dtype="float"), to_categorical(np.array(labels, dtype="float"),C.num_class)

    def deal(self,data):
        image = cv2.imread(C.root + data + "/" + self.mode + ".png")
        image = cv2.resize(image, (C.size[0], C.size[1]), interpolation=cv2.INTER_LINEAR)
        image = augment().tranlate(image)
        return image


#my defined learning strategy
#decay 0.99999 every update, divide 2 when val_loss did not get smaller in 3 times.
class My_learning_rate(Callback):
    def __init__(self,min_lr = 1e-6,max_lr = C.learning_rate,decay=0.99999,monitor='val_loss', cooldown = 2,verbose = 0):
        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.decay = decay
        self.verbose = verbose
        self.monitor = monitor
        self.best = np.Inf
        self.cooldown =cooldown
        self.cooldown_counter = cooldown

    def on_batch_end(self, epoch, logs=None):
        '''Update the learning rate after each batch update'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr,max(self.decay * K.get_value(self.model.optimizer.lr),self.min_lr))

    # decay 0.99999 every update, divide 2 when val_loss did not get smaller in 3 times.
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Reduce LR on plateau conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        else:
            if current<self.best:
                self.best = current
                self.cooldown_counter = self.cooldown
            else:
                if self.in_cooldown():
                    K.set_value(self.model.optimizer.lr,max(0.1 * K.get_value(self.model.optimizer.lr), self.min_lr))
                    print("learning_rate become smaller")
                    self.cooldown_counter = self.cooldown
                else:
                    self.cooldown_counter -= 1

    def in_cooldown(self):
        return self.cooldown_counter < 1

##########################################################
#data augment
##########################################################
class augment():
    def __init__(self):
        self.hehe = 0
        self.direct = [-40,-30,-20,-10,0,10,20,30,40]

    def rd(self):
        a = np.random.randint(0, 9)
        b = np.random.randint(0, 9)
        return a, b

    def tranlate(self,image):
        if np.random.randint(0, 2) == 0:
            image = cv2.flip(image, flipCode=1)
        # a,b= self.rd()
        # rows, cols = image.shape
        # M = np.float32([[1, 0, self.direct[a]], [0, 1, self.direct[b]]])
        # image = cv2.warpAffine(image, M, (cols, rows))
        # label = cv2.warpAffine(label, M, (cols, rows))
        return image

##################################
#for test
##################################
class test_generate():
    def __init__(self):
        self.mode = C.mode

    def deal(self,data):
        image = cv2.imread(C.root + data +"/"+ self.mode + ".png")
        image = cv2.resize(image, (C.size[0], C.size[1]), interpolation=cv2.INTER_LINEAR)
        return image


class cal():
    def __init__(self):
        self.color = ["red","blue"]
        self.classss = ["normal","abnormal"]

    def cal_acc(self,pre,true):
        pre = np.argmax(pre,axis=1)
        return np.mean(np.equal(pre,true))

    def matrix(self,pre,true):
        pre = np.argmax(pre, axis=1)
        matrix = np.zeros((2,2),dtype="float")
        for i in range(len(pre)):
            matrix[int(true[i])][int(pre[i])] += 1
        print(matrix)
        make_matrix(matrix=matrix, name="Confusion_matrix").draw()


    def cal_auc(self,pre,true):
        fpr = []
        tpr = []
        roc_auc = []
        for i in range(C.num_class):
            true_cc = self.get_true(true,i)
            pre_cc = self.get_pre(pre,i)
            a,b,c = self.auc(pre_cc,true_cc)
            fpr.append(a)
            tpr.append(b)
            roc_auc.append(c)

        plt.figure(figsize=(10,10),facecolor='#FFFFFF')
        plt.title("test ROC",fontsize='20')
        for i in range(2):
            if i == 0:
                continue
            plt.plot(fpr[i],tpr[i],label= self.classss[i]+"   auc="+str("%.6f"%roc_auc[i]),c=self.color[i], linestyle="solid", linewidth=3)
            # if i == 1:
                # print("fpr",np.array(fpr[1]))
                # print("tpr",np.array(tpr[1]))
        plt.legend(loc=4, frameon=False, fontsize='28')
        plt.xlim([0,1])
        plt.ylim([0,1.1])
        plt.ylabel("true positive rate",fontsize='30')
        plt.xlabel("false positive rate",fontsize='30')
        plt.show()


    def auc(self,pre,true):
        fpr, tpr, threshold = metrics.roc_curve(true,pre)
        roc_auc = metrics.auc(fpr,tpr)
        return fpr,tpr,roc_auc

    def get_true(self,true,index):
        pp = []
        for i in range(len(true)):
            if true[i] == index:
                pp.append(1)
            else:
                pp.append(0)
        return np.array(pp)

    def get_pre(self,pre,index):
        hehe = []
        for i in range(len(pre)):
            hehe.append(pre[i][index])
        return np.array(hehe)

class make_matrix():
    def __init__(self,matrix,name):
        self.matrix = matrix
        self.classes = ["normal","abnormal"]
        self.classes2 = ["normal","abnormal"]
        self.name = name

    def draw(self):
        plt.figure(facecolor='#FFFFFF', dpi=220)
        self.plot_confusion_matrix(self.matrix,self.classes, normalize=False,
                              title=self.name)

    def plot_confusion_matrix(self,cm,classes,
                              normalize=False,
                              title = None,
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(type(cm))

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, self.classes2, rotation=90, size=18)
        plt.yticks(tick_marks, classes, size=18)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, int(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",size=20)

        plt.tight_layout()
        plt.ylabel('True', size="20")
        plt.xlabel('Predict', size="20")
        plt.show()

class attention():
    def demo(self,att,image,name):
        final = np.mean(att,axis=2)
        final = np.maximum(final,0)
        final /= np.max(final)
        final = cv2.resize(final,(224,224),interpolation=cv2.INTER_LINEAR)
        final = np.uint8(255*final)
        final = cv2.applyColorMap(final,2)
        final = final[:,:,(2,1,0)]
        out = cv2.addWeighted(image,1.0,final,0.3,0)
        # ret, thresh = cv2.threshold(final, C.heat_value, 255, cv2.THRESH_BINARY)
        out = cv2.resize(out,(224,448),interpolation=cv2.INTER_LINEAR)
        plt.figure(figsize=(10,10),facecolor="#FFFFFF")
        plt.imshow(out,cmap="gray")
        plt.axis("on")
        plt.title(name)
        plt.show()

    def siou_for(self,att,rows,cols):
        final = np.mean(att,axis=2)
        final = np.maximum(final,0)
        final /= np.max(final)
        final = cv2.resize(final,(cols,rows),interpolation=cv2.INTER_LINEAR)
        final = np.uint8(255*final)
        ret, thresh = cv2.threshold(final, C.heat_value, 255, cv2.THRESH_BINARY)
        return thresh

    def huatu(self,img,name):
        img = np.uint8(img)
        plt.figure(figsize=(10, 10), facecolor="#FFFFFF")
        plt.imshow(img, cmap="gray")
        plt.axis("on")
        plt.title(name)
        plt.show()