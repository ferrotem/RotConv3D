# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 18:50:09 2020

@author: Utente
"""

import os
import numpy as np
import json
import configAVA as cfg 
from PIL import Image
from tqdm import tqdm
import sys
import gzip
import random

class Loader(object):

    @classmethod
    def load_images(cls, frames_paths_list):
        return np.array([np.array(Image.open(path)) for path in frames_paths_list], dtype=np.uint8)
    
    @classmethod
    def load_batch_images(cls, frames_paths_batch):
        return np.array([Loader.load_images(path_list) for path_list in frames_paths_batch], dtype=np.uint8)
    
    @classmethod            
    def get_frames_and_masks_paths(cls, video_object_json):
        frames_paths_list=[os.path.join(cfg.FRAMES_PATH,video_object_json['v_id'],frame_name+'.png') for frame_name in video_object_json['f_l']]
        masks_paths_list=[[os.path.join(cfg.MASKS_PATH,video_object_json['v_id'],seg_name+'_'+str(person['p_id'])+'.png') for seg_name in video_object_json['f_l'] if os.path.isfile(os.path.join(cfg.MASKS_PATH,video_object_json['v_id'],seg_name+'_'+str(person['p_id'])+'.png'))] for person in video_object_json['p_l']] 
        return frames_paths_list, masks_paths_list
    
    @classmethod
    def check_existence(cls, path_list):
        return len(path_list) == 61 and all(os.path.isfile(path) for path in path_list)
    
    @classmethod
    def check_existence_batch(cls, path_batch):
        return all(Loader.check_existence(path_list) for path_list in path_batch)
    
    @classmethod
    def fill_list_of_list(cls, listt):
        N = max((len(l) for l in listt))
        listt = [l + [-1] * (N - len(l)) for l in listt]; # -1 ignore action
        return listt
    
    @classmethod
    def get_people_ids_actions_bboxes(cls, video_object_json):
        ids=[]
        actions_list = []
        bboxes_list = []
        for person in video_object_json['p_l']:
            ids.append(person['p_id'])
            actions_list.append(person['a_l'])
            bboxes_list.append(person['bb_l'])
        actions_list = Loader.fill_list_of_list(actions_list) # in order to stack
        return np.array(ids,dtype=np.int32), np.array(actions_list,dtype=np.int32), np.array(bboxes_list,dtype=np.float32)
            
    def batch_generator(self): # returns frames, masks, bbox, ids, actions
        for video_object_json in self.video_train:
            frames_paths_list, masks_paths_list = Loader.get_frames_and_masks_paths(video_object_json)
            if Loader.check_existence(frames_paths_list) and Loader.check_existence_batch(masks_paths_list):
                batch_frames = Loader.load_images(frames_paths_list)
                batch_masks = Loader.load_batch_images(masks_paths_list)
                batch_masks = np.swapaxes(batch_masks,0,1)
                ids, actions_list, bboxes_list = Loader.get_people_ids_actions_bboxes(video_object_json)
                bboxes_list = np.swapaxes(bboxes_list,0,1)
                if batch_masks.shape[0]==61 and batch_masks.shape[1]>0 and batch_frames.shape[0]==61:
                    yield np.stack(batch_frames, axis=0), np.stack(batch_masks, axis=0),  np.stack(bboxes_list, axis=0), np.stack(ids, axis=0),  np.stack(actions_list, axis=0) #str(video_object_json)

class Dataset(Loader):
    def __init__(self):
        self.video_train = self.json_loader(cfg.JSON_DATASET)
        self.action_classes = self.json_loader(cfg.CLASSES_PATH)
        self.mask_classes = {0:'background', 1:'person'}    
        random.shuffle(self.video_train)       

    def json_loader(self,json_id):
        with tqdm(total=os.stat(json_id).st_size) as pbar:
            tot_size=0
            def nested_size(obj):
                size = sys.getsizeof(obj)
                if type(obj)==dict:
                    for i in obj.keys():
                        size+=nested_size(obj[i])
                if type(obj)==list:
                    for i in obj:
                        size+=sys.getsizeof(i)
                return size
            def hook(obj):
                nonlocal tot_size
                tot_size+=nested_size(obj)
                pbar.update(tot_size-pbar.n)
                return obj
            with open(json_id) as json_file:
                return json.loads(json_file.read(),object_hook=hook) 
    
