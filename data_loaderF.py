#%%
import os
GPU = "0,1,2"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=GPU

import random
import config as cfg
from PIL import Image
import numpy as np
from utilsF import *
import tensorflow as tf
import matplotlib.pyplot as plt

class Generator(object):
    
    """ this has been changed since now the list of cropped image in list_frames
        is built by the function 
    """
    def _single_input_generator(self, video):
        # print("video: ", video)
        selected_person = random.choice(range( len(self.annotation[video]['p_l']))) # select person from all persons
        label = self._label_generator(video, selected_person)

        org_list, frame_list = self._frame_mask_list_generator(video, selected_person)
        final_input = np.concatenate([frame_list, org_list], axis=-1)
        return final_input, label
    
    ###
               
    """ 
        The following function replaces  _frame_list_generator and _mask_list_generator
    """
    def _frame_mask_list_generator(self, video, selected_person):
        frame_list = []
        org_list = []
        mask_list =[]
        # video =  0 #'qrkff49p4E4'
        frame_id_list =[0,4,9,14,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,44,49,54,59]#np.array(np.arange(60))
        mask_list = []
        # mask_id_list =frame_id_list.copy()#n
        for frame_id  in  frame_id_list:
            frame = self.annotation[video]['f_l'][frame_id]
            # print("frame_id:", frame_id)
            bbox  = self.annotation[video]['p_l'][selected_person]["bb_l"][frame_id]
            v_id =  self.annotation[video]['v_id']
            pathF = os.path.join(cfg.VIDEOS_DATASET_PATH, v_id, frame+'.png' )
            mask_id = frame_id
            
            mask = self.annotation[video]['f_l'][mask_id]
            p_id  = self.annotation[video]['p_l'][selected_person]['p_id']
            pathM = os.path.join(cfg.SEGMENTS_DATASET_PATH, v_id, mask+'_'+str(p_id)+'.png' )
            try:
                img = read_image(pathF )
                mask  = read_image(pathM) 
            except:
                img = Image.new('RGB', (cfg.WIDTH, cfg.HEIGHT), color = (0, 0, 0))
                mask = Image.new('L', (cfg.WIDTH, cfg.HEIGHT), color=0)
            # sng_p, width, height = img_tranfrom(img, bbox)
            sng_p, width, height = img_tranfrom_8_points(img, bbox)
            
            org_img = imm_resize(np.array(img))
            cropped_imm = imm_mask_crop_resize(sng_p, mask) 
            maskx = mask_resize(mask)
            
            """ 
                Now I add noise to the cropped image. You should try both 
                versions: with noise and without noise
                The noise serve to avoid having all zeros in the mask. 
                The noise improves the network ability to find regular patterns (I hope!!!)
            """
            noisy_cropped_imm = add_noise(cropped_imm, maskx)
            
            """ 
                Here note that rescaling the image between 0 and 1 is done ONLY HERE!!!
                To be sure that there are no double normalizations
            """
            org_list.append(org_img/255.0)
        # frame_list.append(cropped_imm/255.0)   ### without added noise 
            frame_list.append(noisy_cropped_imm/255.0) ## with added noise to remove the zeroes
            # mask_list.append(np.array(mask)/1.0)
        return org_list, frame_list#, mask_list
        
###    
    
    """ 
       This is the old function, now replaced by __frame_mask_list_generator 
    """
    # def _frame_list_generator(self, video, selected_person):
    #     frame_list = []
    #     org_list = []
    #     frame_id_list =[0,4,9,14,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,44,49,54,59]#np.array(np.arange(60))
    #     for frame_id in frame_id_list:
    #         frame = self.annotation[video]['f_l'][frame_id]
    #         bbox  = self.annotation[video]['p_l'][selected_person]["bb_l"][frame_id]
    #         v_id =  self.annotation[video]['v_id']
    #         path = os.path.join(cfg.VIDEOS_DATASET_PATH, v_id, frame+'.png' )
    #         try:
    #             img = read_image(path)
    #         except:
    #             img = Image.new('RGB', (cfg.WIDTH, cfg.HEIGHT), color = (0, 0, 0))
    #         # sng_p, width, height = img_tranfrom(img, bbox)
    #         sng_p, width, height = img_tranfrom_8_points(img, bbox)
    #         imgx = imm_resize(sng_p)
    #         org_img = imm_resize(np.array(img))
            
    #         org_list.append(org_img)
    #         frame_list.append(imgx)
    #     return frame_list
    #     frame_list = np.array(frame_list)
    #     org_list = np.array(org_list)
    #     return frame_list, org_list# np.reshape(frame_list,(3,224,224,60))
    # """ This is the old function, now replaced by __frame_mask_list_generator """
    # def _mask_list_generator(self, video, selected_person):
    #     mask_list = []
    #     mask_id_list =[0,4,9,14,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,44,49,54,59]#np.array(np.arange(60))
    #     for mask_id in mask_id_list:
    #         mask = self.annotation[video]['f_l'][mask_id]
    #         bbox  = self.annotation[video]['p_l'][selected_person]["bb_l"][mask_id]
    #         p_id  = self.annotation[video]['p_l'][selected_person]['p_id']
    #         v_id =  self.annotation[video]['v_id']
    #         path = os.path.join(cfg.SEGMENTS_DATASET_PATH, v_id, mask+'_'+str(p_id)+'.png' )
    #         try:
    #             img = read_image(pathF)
    #         except:
    #             img = Image.new('L', (cfg.WIDTH, cfg.HEIGHT), color=0)
        
    #         # sng_p, width, height = img_tranfrom(img, bbox)
    #         imgx = mask_resize(img)
    #         mask_list.append(imgx)
    #     return mask_list
    
    # def combine(frame_lsit, mask_list):
    #     new_list = []
    #     for imm, mask in zip(mask_list, frame_list):
    #         combined =  image_resize(imm, mask)

    # mask_list = np.array(mask_list)
    # return mask_list# np.reshape(mask_list,(3,224,224,60))
    
    

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
        self.train_list, self.val_list = Data_Loader.split_dataset(len(self.annotation))
        self.train_ds = self.initilize_ds(self.train_list)
        self.val_ds = self.initilize_ds(self.val_list)



    @classmethod
    def split_dataset(cls, ds_size):
        total_list = np.arange(ds_size)
        np.random.seed(seed=1717)
        np.random.shuffle(total_list)
        # print("total_list: ", total_list[:3])
        # total_list = total_list[:cfg.DATASET_SIZE]
        # divider =round(cfg.DATASET_SIZE*cfg.SPLIT_RATIO)
        divider =round(ds_size*cfg.SPLIT_RATIO)
        return total_list[:divider-6], total_list[divider+1:]

    @classmethod
    def input_generator(cls, id_list):
        for idx in range(len(id_list)):
            yield id_list[idx]


    
    def read_transform(self, idx):
        [frame_list, label] = tf.py_function(self._single_input_generator, [idx], [tf.float32, tf.int32])
        return frame_list, label

    
    def initilize_ds(self, list_ids):
        ds = tf.data.Dataset.from_generator(Data_Loader.input_generator , args= [list_ids], output_types= (tf.int32))
        ds = ds.map(self.read_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds


# ds = Data_Loader()
# train_ds = ds.train_ds
# # val_ds =ds.val_ds

# for [f,  l] in train_ds.take(2):
#     print(f.shape)
#     print(l)
#     plt.imshow(f[15][:, :, :3].numpy())
    
#     plt.show()
#     plt.imshow(f[15][:, :, 3:].numpy())
#     plt.show()
    
# np.save(f"./results/one_input.npy",f)
# np.save(f"./results/one_input_mask.npy",m)
# for [f, m, l] in val_ds.take(10):
#     print(f.shape)
#     print(m.shape)
#     print(l)
#     plt.imshow(f[0].numpy())
#     plt.show()
#     plt.imshow(m[0].numpy())
#     plt.show()


# %%
