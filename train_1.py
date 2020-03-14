from keras.models import Model
from keras.applications.densenet import DenseNet121
from keras.callbacks import  TensorBoard,ModelCheckpoint
from keras.layers import Dense,Reshape,Input
from keras.optimizers import Adam
from my_things import *
from sklearn.cross_validation import train_test_split
from SIOU import *

def train():
    input_tensor = Input(shape=(C.size[0], C.size[1], 3), name="hehe")
    base_model = DenseNet121(input_tensor=input_tensor, weights='imagenet', include_top=False, pooling='avg')
    x = base_model.output
    predictions = Dense(C.num_class, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    optimizer = Adam(lr=C.learning_rate)
    lr_metric = get_lr(optimizer)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=[lr_metric, "acc"])

    total = prepare_list("all.txt").read_txt()
    print(total)
    train, test = train_test_split(total, random_state=1, train_size=0.85)
    training_generator = DataGenerator_sp(train, batch_size=16)
    validation_generator = DataGenerator_sp(test, batch_size=16)
    checkpointer = ModelCheckpoint(filepath= C.mode+".h5", monitor="val_acc", save_best_only=True, save_weights_only=True)
    ML = My_learning_rate()

    model.fit_generator(training_generator, validation_data=validation_generator, epochs=C.max_epoch, max_queue_size=10,
                        workers=4,callbacks=[checkpointer, ML])

def test():
    input_tensor = Input(shape=(C.size[0], C.size[1], 3), name="hehe")
    base_model = DenseNet121(input_tensor=input_tensor, weights='imagenet', include_top=False, pooling='avg')
    x = base_model.output
    predictions = Dense(C.num_class, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(C.mode+".h5")
    total = prepare_list("all.txt").read_txt()
    train, test = train_test_split(total, random_state=1, train_size=0.85)
    all_pre = None
    all_true = None
    hehe = 0
    for i in range(len(test)):
        if i%100 == 0:
            print(str(i)+"/"+str(len(test)))
        X = test_generate().deal(test[i][0])
        Y = np.array([test[i][1]],dtype="int")
        pre = model.predict([np.array([X/255],dtype="float")])
        if hehe == 0:
            all_pre = pre
            all_true = Y
            hehe = 1
            continue
        all_pre = np.concatenate((all_pre,pre),axis=0)
        all_true = np.concatenate((all_true, Y), axis=0)

    deal = cal()
    print(deal.cal_acc(all_pre,all_true))
    deal.matrix(all_pre,all_true)
    deal.cal_auc(all_pre,all_true)

def attention_map():
    input_tensor = Input(shape=(C.size[0], C.size[1], 3), name="hehe")
    base_model = DenseNet121(input_tensor=input_tensor, weights='imagenet', include_top=False, pooling='avg')
    x = base_model.output
    predictions = Dense(C.num_class, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(C.mode+".h5")
    print("finish load")

    layer_model = Model(inputs=model.input,outputs=model.get_layer("bn").output)
    att = model.get_layer("dense_1").get_weights()[0]
    att = att.transpose((1,0))
    name = ["CR2013ALL^A00^CR-09365133_20131225_0,00010000"]
    for i in range(len(name)):
        pp = name[i].split(",")
        image = test_generate().deal(pp[0])
        nn = int(prepare_list("hehe").deal_label(pp[1]))
        pre = model.predict(np.array([image/255],dtype="float"))
        print(pre,nn)
        catagory_att = att[nn]
        feature = layer_model.predict(np.array([image/255],dtype="float"))[0]
        feature_attention = np.expand_dims(np.expand_dims(catagory_att,axis=0),axis=0)*feature

        attention().demo(feature_attention,image,name[i])

def cal_siou():
    input_tensor = Input(shape=(C.size[0], C.size[1], 3), name="hehe")
    base_model = DenseNet121(input_tensor=input_tensor, weights='imagenet', include_top=False, pooling='avg')
    x = base_model.output
    predictions = Dense(C.num_class, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(C.mode + ".h5")
    print("finish load")

    layer_model = Model(inputs=model.input, outputs=model.get_layer("bn").output)
    att = model.get_layer("dense_1").get_weights()[0]
    att = att.transpose((1, 0))

    total = prepare_list("all.txt").read_txt2()
    train, test = train_test_split(total, random_state=1, train_size=0.85)
    name = total
    # name = ["CR2013ALL^A00^CR-09365133_20131225_0,00010000"]
    ss = Siou()
    pre_correct = 0
    siou_correct = 0
    for i in range(len(name)):
        if i%100 == 0:
            print(str(i)+"/"+str(len(name)))
        pp = name[i].split(",")
        # print(pp[0])
        image = test_generate().deal(pp[0])
        nn = int(prepare_list("hehe").deal_label(pp[1]))

        if nn == 0:
            # print("normal")
            continue
        pre = model.predict(np.array([image / 255], dtype="float"))

        label_pre = np.argmax(pre,axis=1)
        if label_pre != nn:
            # print("predict wrong")
            continue
        pre_correct +=1
        catagory_att = att[nn]
        feature = layer_model.predict(np.array([image / 255], dtype="float"))[0]
        feature_attention = np.expand_dims(np.expand_dims(catagory_att, axis=0), axis=0) * feature
        coordinate = ss.deal_siou(pp[0],feature_attention,pp[1])

        if coordinate == []:
            continue

        print("find the coordinate")
        siou_correct += 1
        if len(coordinate) >1:
            print("############")
            print(pp[0])
        for i in range(len(coordinate)):
            cal_final_coordinate().xml_coordinate(pp[0],coordinate[i],pp[1])
    print("pre_correct",pre_correct,"siou_correct",siou_correct)


if __name__ == '__main__':
    HEHE = 3
    if HEHE == 0:
        train()
    elif HEHE == 1:
        test()
    elif HEHE == 2:
        attention_map()
    else:
        cal_siou()