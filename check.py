import os
import  xml.dom.minidom

def get(root):
    for root, dirs, name in os.walk(root):
        return name

def panduan(root):
    return os.path.exists(root)

def read_coordinate(name,root):
    with open(root + name, "r", encoding="UTF-8") as data:
        object = []
        lines = data.readlines()
        for line in lines:
            wbw = line.split(",")
            # a,b,c,d = wbw[0],wbw[2],wbw[1],wbw[3][:-1]
            a,b,c,d = wbw[0],wbw[1],wbw[2],wbw[3]
            object.append([int(a),int(b),int(c),int(d)])
        return object,len(lines)

def iou(box1, box2, mode):
    '''
    两个框（二维）的 iou 计算

    注意：边框以左上为原点
    box1 predict  box2 man made
    box:[top, left, bottom, right]
    '''
    output = False
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
    inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    sbox1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    sbox2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = sbox1 + sbox2 - inter
    iou = inter / union
    box1_self = inter/sbox1
    box2_self = inter/sbox2
    if mode == "predict" and box1_self > 0.9:
        output = True
    elif mode == "man" and box2_self > 0.9:
        output = True
    elif mode == "iou" and iou>0.1:
        output = True
    return output

def read_xml(id):
    dom = xml.dom.minidom.parse("F:/Annotations/"+id)
    name=dom.getElementsByTagName('name')
    x_min=dom.getElementsByTagName('xmin')
    y_min=dom.getElementsByTagName('ymin')
    x_max=dom.getElementsByTagName('xmax')
    y_max=dom.getElementsByTagName('ymax')
    nodule = []
    count1 = 0
    for i in range(len(name)):
        a=name[i].firstChild.data
        b=x_min[i].firstChild.data
        c=y_min[i].firstChild.data
        d=x_max[i].firstChild.data
        e=y_max[i].firstChild.data
        if a=="nodule":
            nodule.append([float(b),float(c),float(d),float(e)])
            count1 = count1+1
    return nodule,count1

def box_check(txt,xml,mode):
    count = 0
    pp = 0
    for i in range(len(txt)):
        ll =  False
        for j in range(len(xml)):
            if iou(txt[i],xml[j],mode):
                count += 1
                ll = True
        if ll:
            pp += 1

    return count,pp

if __name__ == '__main__':
    name = get("F:/Annotations/")
    # root_txt = "F:/xml_txt3/"
    root_txt = "F:/faster_root/"
    total_box_xml = 0
    total_box_txt = 0
    total_box_right = 0
    predict_right = 0
    for i in range(len(name)):
        name_xml = name[i]
        name_txt = name[i][:-3]+".jpg.txt"
        print(panduan(root_txt+name_txt))
        if not panduan(root_txt+name_txt):
            continue
        xml_coordinate,xml_count = read_xml(name_xml)
        txt_coordinate,txt_count = read_coordinate(name_txt,root_txt)
        total_box_xml += xml_count
        total_box_txt += txt_count
        cc,pp= box_check(txt_coordinate,xml_coordinate,"man")
        total_box_right += cc
        predict_right +=pp

    print("xml_box_total:",total_box_xml,"txt_box_total",total_box_txt)
    print("box_right:",total_box_right)
    print("predict_right",predict_right)