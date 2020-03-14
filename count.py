from random import shuffle

def deal_label(data,mode):
    label = 0
    if mode == "right":
        if int(data[0]) + int(data[2]) + int(data[4]) + int(data[6]) > 0:
            label = 1
    if mode == "left":
        if int(data[1]) + int(data[3]) + int(data[5]) + int(data[7]) > 0:
            label = 1
    if mode == "total":
        if data != "00000000":
            label = 1
    return label

class prepare_list():
    def __init__(self,root):
        self.root = root

    def read_txt(self):
        with open(self.root,"r",encoding="UTF-8") as data:
            name =[]
            lines = data.readlines()
            for line in lines:
                a = line.split(",")[0]
                b = line.split(",")[1]
                name.append([a,b[:-1]])
            shuffle(name)
            return name

a = prepare_list("all.txt").read_txt()
count = 0
for i in range(len(a)):
    count = count+deal_label(a[i][1],"right")

print(str(count)+"/"+str(len(a)))