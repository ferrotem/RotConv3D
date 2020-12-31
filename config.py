# Paths
# VIDEOS_DATASET_PATH = "/home/temir/Documents/AVA/Dataset/Kin200/"
# ANNOTATION_PATH = "/home/temir/Documents/AVA/Dataset/prova.json"

# AVA_path 
VIDEOS_DATASET_PATH = "/media/data4/Datasets/Kinetics_AVA/frames"
# SEGMENTS_DATASET_PATH = "/media/data4/Datasets/Kinetics_AVA/segments"
# ANNOTATION_PATH = "/media/data4/Datasets/Kinetics_AVA/kinetics_frames_boundings_train_v1.0.json" # #ava_frames_boundings_train_v2.2.json


############################### SIAMMASK #####################################à
SEGMENTS_DATASET_PATH = "/media/data4/Datasets/Kinetics_AVA/masks"
ANNOTATION_PATH = "/media/data4/Datasets/Kinetics_AVA/kinetics_frames_masks_train_v1.0.json" # #ava_frames_bmasks_train_v2.2.json

# Input 
NUM_FRAMES= 60
WIDTH = 224
HEIGHT = 224
INPUT_SHAPE_Z= (NUM_FRAMES, WIDTH, HEIGHT, 3+1)
INPUT_SHAPE_X= (WIDTH, HEIGHT, NUM_FRAMES, 3+1)
INPUT_SHAPE_Y= (HEIGHT, NUM_FRAMES, WIDTH, 3+1)

# Network 
NUMBER_OF_RES_BLOCKS = 10
CLASSIFIER = "With_Layer"

DATASET_SIZE = 1000
SPLIT_RATIO = 0.7
DILATION = False 