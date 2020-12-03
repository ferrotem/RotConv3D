# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 00:13:47 2020

@author: Simone Rossetti
"""

import os
import configAVA as cfg

import tensorflow as tf


from matplotlib import pyplot as plt
import numpy as np
autotune = tf.data.experimental.AUTOTUNE

from datasetAVA import Dataset

data  = Dataset()

classes = data.action_classes

train_data = tf.data.Dataset.from_generator(data.batch_generator, output_types= (tf.float32, tf.float32, tf.string))

#images, segs = next(iter(train_data))

for images, segs in train_data.take(1000):
    for i,j in zip(images,np.swapaxes(segs,0,1)):

        # fig=plt.figure(figsize=(10, 10))
        # columns = j.shape[0]+1
        # rows = 1
        # img_batch=[i/255]+[k for k in j]
        # for n in range(1, columns*rows +1):
        #     fig.add_subplot(rows, columns, n)
        #     if n==1:
        #         plt.imshow(img_batch[n-1])
        #     else:
        #         plt.imshow(img_batch[n-1],cmap=plt.get_cmap('hot'), vmin=0, vmax=5)
        #     plt.axis('off')
        # plt.show()
