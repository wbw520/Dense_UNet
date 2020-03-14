import config
import cv2
from my_things import attention
import numpy as np

C = config.Config()

class Siou():
    def __init__(self):
        self.hehe = 0

    def get_x(self, m):
        rows, cols = m.shape
        x1 = 0
        x2 = 0
        change = 0
        for i in range(cols):
            count = 0
            for j in range(rows):
                if m[j][i] != 0:
                    count =  1
            if change == 0:
                if count != 0:
                    x1 = i
                    change = 1
            else:
                if count == 0:
                    x2 = i
                    if (x2 - x1) < C.x_min:
                        change = 0
                        x1 = 0
                        x2 =0
                        continue
                    break
                elif count == 1:
                    x2 = i
        if x1>x2:
            x1 =0
            x2 =0
        return x1, x2

    def get_y(self, m, x1, x2):
        rows, cols = m.shape
        sa = abs(x1 - x2)
        y1 = 0
        y2 = 0
        change = 0
        for i in range(rows):
            count = 0
            for j in range(sa):
                if m[i][x1+j] != 0:
                    count =  1
            if change == 0:
                if count != 0:
                    y1 = i
                    change = 1
            else:
                if count == 0:
                    y2 = i
                    if (y2 - y1) < C.y_min:
                        change = 0
                        y1 = 0
                        y2 = 0
                        continue
                    break
                elif count == 1:
                    y2 = i
        if y1 > y2:
            y1 =0
            y2 =0
        return y1, y2

    def search(self,img):
        x1, x2 = self.get_x(img)
        y1, y2 = self.get_y(img, x1, x2)
        hehe = True
        if (x1 + x2 + y1 + y2) != 0:
            for i in range(x1,x2+1):
                for j in range(y1,y2+1):
                    img[j][i] = 0
        else:
            hehe = False
        return [x1,x2,y1,y2],img,hehe

    def search2(self,img):
        x1, x2 = self.get_x(img)
        y1, y2 = self.get_y(img, x1, x2)
        return [x1,x2,y1,y2]

    def deep_search(self,img):
        # print("start deep search")
        coordinate = []
        hehe = True
        while hehe:
            coordi,img,hehe = self.search(img)
            if hehe:
                coordinate.append(coordi)
        # print("deep search finish")
        return coordinate

    def location_discriminate(self,location):
        location_have = []
        if C.mode == "right":
            LL = [0,2,4,6]
            PP = [5,6,7,8]
            for i in range(len(LL)):
                if location[LL[i]] != "0":
                    location_have.append(PP[i])
        elif C.mode == "left":
            LL = [1,3,5,7]
            PP = [4,3,2,1]
            for i in range(len(LL)):
                if location[LL[i]] != "0":
                    location_have.append(PP[i])
        return location_have

    def siou(self,att,mask,condition,coordinate):
        rows, cols = mask.shape
        self_count = 0
        iou_count = 0
        hehe = False
        new_coordinate = 0
        PP = np.zeros((rows,cols),dtype="int")
        for i in range(coordinate[0], coordinate[1]+1):
            for j in range(coordinate[2], coordinate[3]+1):
                if att[j][i] == 255:
                    self_count += 1
                    if mask[j][i] in condition:
                        iou_count += 1
                        PP[j][i] = 255
        standard = iou_count/(self_count+1)
        # print("siou for this patch",standard)
        if standard>C.standard:
            hehe = True
            new_coordinate = self.search2(PP) #only use coordinate which in siou
        return hehe,new_coordinate

    def deal_siou(self,name,att_map,label):
        correct_coordinate = []
        location_condition = self.location_discriminate(label)
        mask = cv2.imread(C.root + name + "/" + C.mode + "_mask.png", cv2.IMREAD_GRAYSCALE)
        rows, cols = mask.shape
        att_resized = attention().siou_for(att_map, rows,cols)
        # attention().huatu(att_resized, "0")
        coordinate = self.deep_search(att_resized)
        att_re = attention().siou_for(att_map, rows, cols)
        for i in range(len(coordinate)):
            outcome,new_coordinate = self.siou(att_re,mask,location_condition,coordinate[i])
            if outcome:
                correct_coordinate.append(new_coordinate)
        return correct_coordinate

class cal_final_coordinate():
    def __init__(self):
        self.mode = C.mode

    def read_txt_coordinate(self,name):
        with open(C.root + name + "/coordinate.txt","r",encoding="UTF-8") as data:
            name =[]
            lines = data.readlines()
            for line in lines:
                a = line.split(",")
                name.append(a)
            if self.mode == "right":
                pp = 0
            else:
                pp =1
            return name[pp]

    def translate_coordinate(self,zuobiao,target,orl):
        rows_orl, cols_orl = orl.shape
        rows_target, cols_target = target.shape
        new_x1 = int(cols_target * zuobiao[0] / cols_orl)
        new_x2 = int(cols_target * zuobiao[1] / cols_orl)
        new_y1 = int(rows_target * zuobiao[2] / rows_orl)
        new_y2 = int(rows_target * zuobiao[3] / rows_orl)
        return [new_x1,new_x2,new_y1,new_y2]

    def total_image_coordinate(self,coordinate_seg_att,coordinate_seg_box):
        coordinate_seg_att[0] += int(coordinate_seg_box[1])
        coordinate_seg_att[1] += int(coordinate_seg_box[1])
        coordinate_seg_att[2] += int(coordinate_seg_box[3])
        coordinate_seg_att[3] += int(coordinate_seg_box[3])
        return coordinate_seg_att

    def write_txt(self,coordinate,name):
        hehe = open(C.xml_root + name + ".txt", "a")
        wbw_left = str(coordinate[0]) + "," + str(coordinate[1]) + "," +str(coordinate[2]) + "," +str(coordinate[3])+"\n"
        hehe.write(wbw_left)


    def xml_coordinate(self,name,att_coordinate,label):
        mask = cv2.imread(C.root + name + "/" + C.mode + "_mask.png", cv2.IMREAD_GRAYSCALE)
        seg_image = cv2.imread(C.root + name + "/" + C.mode + ".png", cv2.IMREAD_GRAYSCALE)
        total_image = cv2.imread(C.root + name + "/" + "total.png", cv2.IMREAD_GRAYSCALE)
        seg_box_coordinate = self.read_txt_coordinate(name)
        seg_att_coordinate = self.translate_coordinate(att_coordinate,seg_image,mask)
        total_image_coordi = self.total_image_coordinate(seg_att_coordinate,seg_box_coordinate)
        xml_size = np.zeros((1000,1000),dtype="int")
        xml_coordinate = self.translate_coordinate(total_image_coordi,xml_size,total_image)
        self.write_txt(xml_coordinate,name)
        xml_image = cv2.resize(total_image, (1000, 1000), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(C.xml_root +name + ".png",xml_image)




