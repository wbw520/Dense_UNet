import  xml.dom.minidom
import os
import numpy as np
import matplotlib.pyplot as plt
from decimal import *
from matplotlib.patches import ConnectionPatch
a = [0.769763, 0.722167, 0.84983635, 0.84198, 0.92170715, 0.9197626, 0.90055656, 0.9049797, 0.9050083, 0.9432001, 0.941947, 0.9382191, 0.9439411, 0.94146156, 0.9451084, 0.9456539, 0.9466591, 0.9468422, 0.94680595, 0.94748497]
b = [0.5328064, 0.47922516, 0.5681553, 0.7198973, 0.56908417, 0.60331726, 0.5795927, 0.8627291, 0.91136265, 0.91371346, 0.935092, 0.93802166, 0.94861984, 0.94689655, 0.9495155, 0.9508475, 0.9508494, 0.9510472, 0.95134273, 0.95123905]
c = [0.6007347, 0.56949997, 0.7835808, 0.56947136, 0.58673, 0.5765581, 0.5941353, 0.7439842, 0.8374653, 0.91968536, 0.93836975, 0.93201065, 0.92886543, 0.9354849, 0.9423614, 0.9432907, 0.943502, 0.9433189, 0.9433748, 0.94322124]
d = [0.53358555, 0.56987286, 0.72931767, 0.59825134, 0.7972002, 0.78339005, 0.8175459, 0.85843945, 0.79659176, 0.79565144, 0.8915224, 0.8655634, 0.8920469, 0.8621397, 0.88037777, 0.881217, 0.8864393, 0.8858099, 0.883667, 0.89305305]

epoch = range(20)

plt.figure(figsize=(16,8),dpi=98,facecolor='#FFFFFF')
ax1 = plt.subplot(121)
p2 = plt.subplot(122)

ax1.plot(epoch, c, linewidth=3, label="DU-Net+Focal Loss (384x384)", c="green",linestyle="solid")
ax1.plot(epoch, b, linewidth=3, label="DU-Net+Focal Loss", c="red",linestyle="solid")
ax1.plot(epoch, a, linewidth=3, label="DU-Net+Cross Entropy", c="blue",linestyle="solid")
ax1.plot(epoch, d, linewidth=3, label="DU-Net++dice", c="yellow",linestyle="solid")
ax1.set_xlabel('epoch', size="35")
ax1.set_ylabel('Pixel Accuracy', size="32")
plt.tick_params(labelsize=18)
ax1.set_xlim([0, 19])
ax1.set_ylim([0, 1])
ax1.legend(loc=4,frameon=False,fontsize='23')


p2.plot(epoch, c, linewidth=3, label="DU-Net+Focal Loss (384x384)", c="green",linestyle="solid")
p2.plot(epoch, b, linewidth=3, label="DU-Net+Focal Loss", c="red",linestyle="solid")
p2.plot(epoch, a, linewidth=3, label="DU-Net+Cross Entropy", c="blue",linestyle="solid")
p2.plot(epoch, d, linewidth=3, label="DU-Net++dice", c="yellow",linestyle="solid")


p2.axis([13,18,0.8,1])
p2.set_xlabel('epoch',fontsize=25)
p2.legend()

tx0 = 13
tx1 = 18
ty0 = 0.81
ty1 = 0.98
sx = [tx0,tx1,tx1,tx0,tx0]
sy = [ty0,ty0,ty1,ty1,ty0]
ax1.plot(sx,sy,"purple", linewidth=3)

xy = (18, 0.98)
xy2 = (13, 0.99)
con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                      axesA=p2, axesB=ax1, linewidth=3)
p2.add_artist(con)

xy = (18, 0.81)
xy2 = (13,0.8)
con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                      axesA=p2, axesB=ax1, linewidth=3)
p2.add_artist(con)

# plt.savefig('PR')
plt.show()