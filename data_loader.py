#%%
import os
import random
import config as cfg
from PIL import Image
import numpy as np
from utils import *
import tensorflow as tf
import matplotlib.pyplot as plt
#%%
class Generator(object):
    
    def _single_input_generator(self, video):
        #video = self.id_list[index]
        selected_person = random.choice(range( len(self.annotation[video]['p_l']))) # select person from all persons
        
        label = self._label_generator(video, selected_person)
        frame_list = self._frame_list_generator (video, selected_person)
        return frame_list, label
    
    def _frame_list_generator(self, video, selected_person):
        frame_list = []
        frame_id_list =np.array(np.arange(60))
        for frame_id in frame_id_list:
            frame = self.annotation[video]['f_l'][frame_id]
            bbox  = self.annotation[video]['p_l'][selected_person]["bb_l"][frame_id]
            v_id =  self.annotation[video]['v_id']
            path = os.path.join(cfg.VIDEOS_DATASET_PATH, v_id, frame+'.png' )
            
            img = read_image(path)
            sng_p, width, height = img_tranfrom(img, bbox)
            imgx = imm_resize(sng_p, width, height)
            frame_list.append(imgx)

        frame_list = np.array(frame_list)
        return frame_list# np.reshape(frame_list,(3,224,224,60))
    
    def _label_generator(self, video, selected_person):
        action_list = self.annotation[video]['p_l'][selected_person]['a_l']
        label= np.zeros(80)
        for action in action_list:
             label[action-1]=1
        return label


class Data_Loader(Generator):
    def __init__(self):
        self.annotation = file_reader(cfg.ANNOTATION_PATH)
        self.train_list, self.val_list = Data_Loader.split_dataset(len(self.annotation))
        self.train_ds = self.initilize_ds(self.train_list)
        self.val_ds = self.initilize_ds(self.val_list)



    @classmethod
    def split_dataset(cls, ds_size):
        total_list = np.arange(ds_size)
        np.random.shuffle(total_list)
        divider =round(ds_size*cfg.SPLIT_RATIO)
        return total_list[:divider], total_list[divider:]

    @classmethod
    def input_generator(cls, id_list):
        for idx in range(len(id_list)):
            yield id_list[idx]

    
    def read_transform(self, idx):
        [frame_list, label] = tf.py_function(self._single_input_generator, [idx], [tf.float32, tf.int32])
        return frame_list, label

    
    def initilize_ds(self, list_ids):
        ds = tf.data.Dataset.from_generator(Data_Loader.input_generator , args= [list_ids], output_types= (tf.int32))
        ds = ds.map(self.read_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds

#%%
ds = Data_Loader()
train_ds = ds.train_ds
val_ds =ds.val_ds
# %%
for [f, l] in train_ds.take(2):
    print(f.shape)
    print(l)
    plt.imshow(f[0].numpy())
    break


for [f, l] in val_ds.take(2):
    print(f.shape)
    print(l)
    plt.imshow(f[0].numpy())
    