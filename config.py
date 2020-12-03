# Paths
VIDEOS_DATASET_PATH = "/home/temir/Documents/AVA/Dataset/Kin200/"
ANNOTATION_PATH = "/home/temir/Documents/AVA/Dataset/prova.json"

# AVA_path 
VIDEOS_DATASET_PATH = "/media/data4/Datasets/AVA_Kinetics_frames"
ANNOTATION_PATH = "/media/data4/Datasets/Kinetics_AVA/ava_frames_boundings_train_v2.2.json"

# Input 
NUM_FRAMES= 60
WIDTH = 224
HIGHT = 224
INPUT_SHAPE_Z= (NUM_FRAMES, WIDTH, HIGHT, 3)
INPUT_SHAPE_X= (WIDTH, HIGHT, NUM_FRAMES, 3)
INPUT_SHAPE_Y= (HIGHT, NUM_FRAMES, WIDTH, 3)

# Network 
NUMBER_OF_RES_BLOCKS = 10
CLASSIFIER = "Dilation_2"

SPLIT_RATIO = 0.7
DILATION = True