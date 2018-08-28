from __future__ import absolute_import, division, print_function

import tensorflow as tf

tf.enable_eager_execution()

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import time

data_dir = 'data'
data_file = 'commentPairs.txt'

path_to_file = os.path.join(data_dir, '/', data_file)
print(path_to_file)
