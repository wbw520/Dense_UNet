import matplotlib.pyplot as plt
import os
import cv2

def huatu(img,name,coordinate):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, aspect='equal')
    for i in range(len(coordinate)):
        bbox = coordinate[i]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
    ax.set_title(name, fontsize=14)
    plt.tight_layout()
    plt.draw()
    plt.savefig("hehe/" + name)

def get2(root):
    for root, dirs, SB in os.walk(root):
        return SB

def read_coordinate(name,root):
    with open(root + name + ".txt", "r", encoding="UTF-8") as data:
        object = []
        lines = data.readlines()
        for line in lines:
            wbw = line.split(",")
            a,b,c,d = wbw[0],wbw[2],wbw[1],wbw[3][:-1]
            object.append([int(a),int(b),int(c),int(d)])
        return object

if __name__ == '__main__':
    root = "F:/xml_txt/"
    root_image = "F:/xml_data/"
    root_for_patch = "F:/patch_file/"
    all_txt = get2(root)
    for i in range(len(all_txt)):
        name = all_txt[i][:-4]
        print(name)
        coordinate = read_coordinate(name,root)
        print(coordinate)
        img = cv2.imread(root_image+name+".png")
        huatu(img,name+".png",coordinate)
