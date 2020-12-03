# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 11:41:42 2020

@author: Utente
"""


GPU = '0'
#CLASSIFIER = 'encoder'+GPU
#N_CLASSES = 8
#TRAIN_SIZE = 10000
#TEST_SIZE = 10
#BUFFER_SIZE = 400
#INIT_LR = 1e-04
BATCH_SIZE = 61
#EPOCHs = 50
#ENHANCE = False
#NUM_FRAMES= 2 #CHANGE_WIDTH = True
#DROPOUT_RATE = 0.2
#NUMBER_OF_VIDEOS = 124
#N_STEP =int(TRAIN_SIZE/BATCH_SIZE)
IMAGE_WIDTH = 512
#INPUT_SHAPE= (NUM_FRAMES, IMAGE_WIDTH, IMAGE_WIDTH, 3)
#MARGIN = 0.5
#ALPHA = 0.01

PATH = "/media/data4/Datasets/Kinetics_AVA/"
    
JSON_DATASET = PATH + 'ava_frames_masks_train_v2.2.json'

FRAMES_PATH = PATH + 'frames/' 

MASKS_PATH = PATH + "masks/"

CLASSES_PATH = PATH + "ava_action_list_v2.2.json"