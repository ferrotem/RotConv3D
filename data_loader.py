#%%
import os
import random
import config as cfg
from PIL import Image
import numpy as np
from utils import *
import tensorflow as tf
#%%
class Dataset():
    def __init__(self, annotation, id_list):
        self.annotation = annotation
        self.id_list = id_list
    

    def _input_generator(self):
        for idx in range(len(self.id_list)):
            yield self.id_list[idx]
    

    def _single_input_generator(self, index):
        video = self.id_list[index]
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
            v_id = self.annotation[video]['v_id']
            path = os.path.join(cfg.VIDEOS_DATASET_PATH, v_id, frame+'.png' )
            
            img = read_image(path)
            sng_p, width, height = img_tranfrom(img, bbox)
            imgx = imm_resize(sng_p, width, height)
            frame_list.append(imgx)

        frame_list = np.array(frame_list)
        return np.reshape(frame_list,(3,224,224,60))
    
    def _label_generator(self, video, selected_person):
        action_list = self.annotation[video]['p_l'][selected_person]['a_l']
        label= np.zeros(80)
        for action in action_list:
             label[action-1]=1
        return label


#%%
annotation = file_reader(cfg.ANNOTATION_PATH)
data_ratio = 0.7
total_list = np.arange(len(annotation))
np.random.shuffle(total_list)
divider =round(len(annotation)*data_ratio)
train_list, val_list = total_list[:divider], total_list[divider:]
#%%
train_loader = Dataset(annotation, train_list)
val_loader = Dataset(annotation, val_list)
#%%
for i in dataset._input_generator():
    print(i)
    break
# %%
train_ds = tf.data.Dataset.from_generator(train_loader._input_generator , output_types= (tf.int32))
val_ds = tf.data.Dataset.from_generator(val_loader._input_generator , output_types= (tf.int32))

# autotune = tf.data.experimental.AUTOTUNE
# train_ds = train_ds.prefetch(autotune)
# %%
dataset = [image,] = tf.py_function(random_rotate_image, [image], [tf.float32])

# %%
