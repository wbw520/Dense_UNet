from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from model import *
from tool_function import *
from load_data import *
from focal_loss import *

batch_size = 8
epoch_max = 100
epoch_length = 250
learning_rate = 0.001

input_u = Input(shape=(256,256,1))
out = DU_Net(input_u,0.5)
# out = unet(input_u)
model_U_net = Model(input_u,out)
optimizer = Adam(lr=learning_rate)
lr_metric = get_lr(optimizer)
model_U_net.compile(optimizer=optimizer,loss=dice_coe,metrics=['accuracy',lr_metric])
#"categorical_crossentropy"


print("load data")
load_name()
print("start training")
ACC = []
for epoch_num in range(epoch_max):
    iteration = 0
    if LR(epoch_num):
        K.set_value(optimizer.lr,0.1*K.get_value(optimizer.lr))
    while iteration < epoch_length:
        print("epoch:" + str(epoch_num) + "/" + str(epoch_max),"iteration:" + str(iteration) + "/" + str(epoch_length))
        X,Y = get_batch_train(batch_size)
        out1 = model_U_net.train_on_batch(X,Y)
        print(out1)
        iteration = iteration+1
    print("demo")
    demo_in,demo_origin = get_demo()
    demo = model_U_net.predict([demo_in])
    demo = np.argmax(demo, axis=3)
    show_demo(demo,demo_origin)

    print("-----val-----")
    T_X,T_Y = get_batch_test(16)
    pre_X = model_U_net.predict(T_X)
    acc = model_U_net.test_on_batch(T_X,T_Y)
    ACC.append(acc[1])
    pre_T = np.argmax(pre_X,3)
    label_T = np.argmax(T_Y,3)
    IOU = iou(pre_T,label_T,16)
    print(acc)
    print("tip_iou", IOU[0], "up_iou", IOU[1], "middle_iou", IOU[2], "lower_iou", IOU[3])
    print(ACC)