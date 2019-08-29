from load_data import *
import numpy as np
from keras.utils import to_categorical

load_name()
X,Y = get_batch_train(8)

