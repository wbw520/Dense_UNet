import tensorflow as tf
import numpy as np
from keras import backend as K

sess=tf.Session()
a = tf.constant(np.ones((3,2,2),dtype="float"),dtype=tf.float32)
b = tf.constant([[1,1],[1,0]],dtype=tf.float32)
c = tf.constant(1,dtype=tf.float32)
p = K.equal(a,b)
p = K.cast(p,dtype=tf.float32)
s = K.sum(p,axis=(1,2))
print(sess.run(s))
print(sess.run(c))
print(sess.run(tf.equal(b,c)))