import os
import json
from PIL import Image
import numpy as np
import config as cfg

def file_reader(file_name):
    with open(file_name) as json_file:
        return json.load(json_file)


def file_writer(file, file_name):
    with open(file_name, 'w') as f:
        f.write(json.dumps(file)) 


def imm_resize(img, width, height):
    imgn = Image.fromarray(img.astype(np.uint8))
    imgx = imgn.resize((224,224), Image.ANTIALIAS)
    return np.array(imgx)/255

def img_tranfrom(img, bbox):
    width, height = img.size
    T = [width, height, width, height]
    mask = np.zeros((height,width))
    bbox= np.array(bbox)
    xmin, ymin, xmax, ymax = (bbox*T)
    mask[int(round(ymin)): int(round(ymax)), int(round(xmin)) : int(round(xmax))] = 1
    single_person = np.multiply(np.array(img),mask[:,:, np.newaxis])
    return single_person.astype(int) ,width, height

def read_image(img_path):
    return Image.open(img_path)