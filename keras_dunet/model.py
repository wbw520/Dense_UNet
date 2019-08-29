from keras.layers import Activation,Convolution2D,MaxPool2D,Dropout,Concatenate,Deconvolution2D,UpSampling2D
from keras.layers.normalization import BatchNormalization as BN
from keras import backend as K
import numpy as np
from keras.regularizers import l2


#------------------------------DU-Net--------------------------------------------------

def DU_Net(input,rate,trainable=True):
    n1 = conv_block(input,stage=1,filters=32,rate=rate,trainable=trainable)
    m1 = MaxPool2D(pool_size=(2,2))(n1)
    n2 = conv_block(m1, stage=2, filters=64, rate=rate, trainable=trainable)
    m2 = MaxPool2D(pool_size=(2, 2))(n2)
    n3 = conv_block(m2, stage=3, filters=128, rate=rate, trainable=trainable)
    m3 = MaxPool2D(pool_size=(2, 2))(n3)
    n4 = conv_block(m3, stage=4, filters=256, rate=rate, trainable=trainable)
    m4 = MaxPool2D(pool_size=(2, 2))(n4)
    n5 = conv_block(m4, stage=5, filters=512, rate=rate, trainable=trainable)
    m5 = MaxPool2D(pool_size=(2, 2))(n5)

    n_end = conv_block(m5,stage=6,filters=1024,rate=rate,trainable=trainable)

    up1 = up(n_end,n5,1, 512,rate=rate,trainable=trainable)
    up2 = up(up1, n4, 2, 256, rate=rate, trainable=trainable)
    up3 = up(up2, n3, 3, 128, rate=rate, trainable=trainable)
    up4 = up(up3, n2, 4, 64, rate=rate, trainable=trainable)
    up5 = up(up4, n1, 5, 32, rate=rate, trainable=trainable)

    final = Convolution2D(5,(1,1),name="output",padding="SAME",kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4),trainable=trainable)(up5)
    final = Activation("softmax",name="final")(final)
    return final



def conv_block(input,stage,filters,rate,state="down",trainable = True):
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = state + 'conv' + str(stage) + '_branch'
    bn_name_base = state + 'bn' + str(stage) + '_branch'
    x = Convolution2D(filters,(1,1),name=conv_name_base+"a",padding="SAME",
                      kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4),trainable=trainable)(input)
    x = Activation('relu')(x)
    x = BN(axis=bn_axis,name=bn_name_base+"a",gamma_regularizer=l2(1e-4),beta_regularizer=l2(1e-4))(x)

    x = Dropout(rate=rate)(x)

    x = Convolution2D(filters, (3, 3), name=conv_name_base + "b", padding="SAME",
                      kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4),trainable=trainable)(x)
    x = Activation('relu')(x)
    x = BN(axis=bn_axis, name=bn_name_base + "b",gamma_regularizer=l2(1e-4),beta_regularizer=l2(1e-4))(x)
    out = Concatenate(axis=bn_axis)([input,x])
    return out

def up(input,conta,stage,filters,rate,trainable = True):
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    upsampling = "de" + str(stage)
    bn_name_base = "de_" + 'bn' + str(stage) + '_branch'
    x = Deconvolution2D(filters,(3,3),strides=(2,2),padding="SAME",name=upsampling,
                        kernel_regularizer=l2(1e-4),bias_regularizer=l2(1e-4),trainable=trainable)(input)
    # x = BN(axis=bn_axis, name=bn_name_base,gamma_regularizer=l2(1e-4),beta_regularizer=l2(1e-4))(x)
    # x = Activation('relu')(x)
    con = Concatenate(axis=bn_axis)([x,conta])
    out = conv_block(con,stage=stage,filters=filters,rate=rate,state="up",trainable=trainable)
    return out


#------------------------------U-Net--------------------------------------------------

def unet(input):
    conv1 = Convolution2D(64,(3, 3), activation='relu', padding='same')(input)
    conv1 = Convolution2D(64,(3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPool2D(pool_size=(2,2))(conv1)
    conv2 = Convolution2D(128,(3, 3), activation='relu', padding='same')(pool1)
    conv2 = Convolution2D(128,(3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPool2D(pool_size=(2,2))(conv2)
    conv3 = Convolution2D(256,(3, 3), activation='relu', padding='same')(pool2)
    conv3 = Convolution2D(256,(3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPool2D(pool_size=(2,2))(conv3)
    conv4 = Convolution2D(512,(3, 3), activation='relu', padding='same')(pool3)
    conv4 = Convolution2D(512,(3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2))(conv4)
    conv5 = Convolution2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Convolution2D(1024, (3, 3), activation='relu', padding='same')(conv5)

    up1 = Concatenate(axis=3)([UpSampling2D(size=(2,2))(conv5),conv4])
    conv6 = Convolution2D(512,(3, 3), activation='relu', padding='same')(up1)
    conv6 = Convolution2D(512, (3, 3), activation='relu', padding='same')(conv6)
    up2 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Convolution2D(256, (3, 3), activation='relu', padding='same')(up2)
    conv7 = Convolution2D(256, (3, 3), activation='relu', padding='same')(conv7)
    up3 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Convolution2D(128, (3, 3), activation='relu', padding='same')(up3)
    conv8 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv8)
    up4 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Convolution2D(64, (3, 3), activation='relu', padding='same')(up4)
    conv9 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv9)

    final = Convolution2D(5, (1, 1))(conv9)
    final = Activation("softmax", name="final")(final)

    return final






