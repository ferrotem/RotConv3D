#%%
import os
GPU = "0"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=GPU
#%%
import random
import config as cfg
from PIL import Image
import numpy as np
from utils import *
import tensorflow as tf
import matplotlib.pyplot as plt


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
#%%
class Generator(object):
    
    def _single_input_generator(self, video):
        # video = self.id_list[index]
        selected_person = random.choice(range( len(self.annotation[video]['p_l']))) # select person from all persons
        # video = self.listen_class[idx]['video']
        # selected_person = self.listen_class[idx]['selected_person']
        label = self._label_generator(video, selected_person)
        # label= np.zeros(80)
        # while label[36]!=1:
        #     selected_person = random.choice(range( len(self.annotation[video]['p_l']))) # select person from all persons
        #     label = self._label_generator(video, selected_person)
        frame_list = self._frame_list_generator (video, selected_person)
        mask_list = self._mask_list_generator (video, selected_person)
        return frame_list, mask_list, label
    
    def _frame_list_generator(self, video, selected_person):
        frame_list = []
        frame_id_list =np.array(np.arange(60))
        for frame_id in frame_id_list:
            frame = self.annotation[video]['f_l'][frame_id]
            bbox  = self.annotation[video]['p_l'][selected_person]["bb_l"][frame_id]
            v_id =  self.annotation[video]['v_id']
            path = os.path.join(cfg.VIDEOS_DATASET_PATH, v_id, frame+'.png' )
            try:
                img = read_image(path)
            except:
                img = Image.new('RGB', (cfg.WIDTH, cfg.HEIGHT), color = (0, 0, 0))
            # sng_p, width, height = img_tranfrom(img, bbox)
            sng_p, width, height = img_tranfrom_8_points(img, bbox)
            imgx = imm_resize(sng_p, width, height)
            frame_list.append(imgx)

        frame_list = np.array(frame_list)
        return frame_list# np.reshape(frame_list,(3,224,224,60))
    
    def _mask_list_generator(self, video, selected_person):
        mask_list = []
        mask_id_list =np.array(np.arange(60))
        for mask_id in mask_id_list:
            mask = self.annotation[video]['f_l'][mask_id]
            bbox  = self.annotation[video]['p_l'][selected_person]["bb_l"][mask_id]
            p_id  = self.annotation[video]['p_l'][selected_person]['p_id']
            v_id =  self.annotation[video]['v_id']
            path = os.path.join(cfg.SEGMENTS_DATASET_PATH, v_id, mask+'_'+str(p_id)+'.png' )
            try:
                img = read_image(path)
            except:
                img = Image.new('L', (cfg.WIDTH, cfg.HEIGHT), color=0)
            # sng_p, width, height = img_tranfrom(img, bbox)
            imgx = mask_resize(img)
            mask_list.append(imgx)

        mask_list = np.array(mask_list)
        return mask_list# np.reshape(mask_list,(3,224,224,60))

    def _label_generator(self, video, selected_person):
        action_list = self.annotation[video]['p_l'][selected_person]['a_l']
        action_list = list(set(action_list))
        label= np.zeros(80)
        for action in action_list:
            # if action==37 or action==4:
            label[action-1]=1
        return label


class Data_Loader(Generator):
    def __init__(self):
        self.annotation = file_reader(cfg.ANNOTATION_PATH)
        # self.listen_class = file_reader("class_4_37.json")
        ## self.total_list = list(self.listen_class.keys())
        self.train_list, self.val_list = Data_Loader.split_dataset(len(self.annotation))
        self.train_ds = self.initilize_ds(self.train_list)
        self.val_ds = self.initilize_ds(self.val_list)



    @classmethod
    def split_dataset(cls, ds_size):
        total_list = np.arange(ds_size)
        np.random.seed(seed=1717)
        np.random.shuffle(total_list)
        total_list = total_list[:cfg.DATASET_SIZE]
        divider =round(cfg.DATASET_SIZE*cfg.SPLIT_RATIO)
        return total_list[:divider], total_list[divider:]

    @classmethod
    def input_generator(cls, id_list):
        for idx in range(len(id_list)):
            yield id_list[idx]


    
    def read_transform(self, idx):
        [frame_list, masks_list, label] = tf.py_function(self._single_input_generator, [idx], [tf.float32, tf.float32, tf.int32])
        return frame_list, masks_list, label

    
    def initilize_ds(self, list_ids):
        ds = tf.data.Dataset.from_generator(Data_Loader.input_generator , args= [list_ids], output_types= (tf.int32))
        ds = ds.map(self.read_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds

#%%
# ds = Data_Loader()
# train_ds = ds.train_ds
# val_ds =ds.val_ds
# # %%
# for [f, m, l] in train_ds.take(10):
#     print(f.shape)
#     print(m.shape)
#     print(l)
#     plt.imshow(f[0].numpy())
#     plt.show()
#     plt.imshow(m[1].numpy())
#     plt.show()
    


# for [f, m, l] in val_ds.take(10):
#     print(f.shape)
#     print(m.shape)
#     print(l)
#     plt.imshow(f[0].numpy())
#     plt.show()
#     plt.imshow(m[0].numpy())
#     plt.show()

# %%
