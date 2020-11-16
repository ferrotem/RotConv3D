import os
import torch
import random
import config as cfg
from PIL import Image
import numpy as np
from utils import *


class Dataset():
    def __init__(self, annotation, id_list):
        self.annotation = annotation
        self.id_list = id_list
    
    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        video = self.id_list[index]
        # print("video: ", video)
        num_people = len(self.annotation[video]['p_l'])
        selected_person = random.choice(range(num_people))
        # selected_person = 0
        action_list = self.annotation[video]['p_l'][selected_person]['a_l']
        label= np.zeros(80)
        for action in action_list:
             label[action-1]=1
        # print(label)
        frame_id_list =np.array(np.arange(60)) #[0] + random.sample(range(1,28),4) + [29] + random.sample(range(30,60),4)+[60]
        frame_list = []
        # print(len(frame_id_list))
        for frame_id in frame_id_list:
            frame = self.annotation[video]['f_l'][frame_id]
            bbox  = self.annotation[video]['p_l'][selected_person]["bb_l"][frame_id]
            v_id = self.annotation[video]['v_id']
            path = os.path.join(cfg.VIDEOS_DATASET_PATH, v_id, frame+'.png' )
            img = read_image(path)
            sng_p, width, height = img_tranfrom(img, bbox)
            imgx = imm_resize(sng_p, width, height)
            # print(imgx.size)
            frame_list.append(imgx)

        frame_list = torch.tensor(frame_list)
        # perm = torch.LongTensor([0,4,2,3,1])
        
        return torch.reshape(frame_list,(3,224,224,60)),  label