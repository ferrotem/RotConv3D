import os
import json
from PIL import Image,ImageDraw
import numpy as np
import config as cfg
import pickle
from skimage.util import img_as_ubyte
from skimage import  img_as_float
def file_reader(file_name):
    with open(file_name) as json_file:
        return json.load(json_file)


def file_writer(file, file_name):
    with open(file_name, 'w') as f:
        f.write(json.dumps(file)) 

def imm_resize(img):
    imgn = Image.fromarray(img.astype(np.uint8))
    imgx = imgn.resize((cfg.WIDTH,cfg.WIDTH), Image.ANTIALIAS)
    return np.array(imgx)

""" This is a new function that crop the image while resizing it """
def imm_mask_crop_resize(img, mask):
    imgn = Image.fromarray(img.astype(np.uint8))
    imgx = imgn.resize((cfg.WIDTH,cfg.WIDTH), Image.ANTIALIAS)
    img_uint8 = np.array(imgx)
    maskx = mask.resize((cfg.WIDTH,cfg.WIDTH), Image.ANTIALIAS)
    maskx = np.array(maskx)
    cropped_imm = np.zeros_like(img_uint8)
    for kk in range(3):
        cropped_imm[:,:,kk] = img_uint8[:,:,kk]*maskx
        
    return cropped_imm

""" This is a new function that add noise to the zeros in the mask """
def add_noise(cropped_img, maskx):
    masku = np.array(maskx)
    A = 1 - masku
    cimm = cropped_img.astype(np.float)
    # U = np.random.rand(masku.shape[0],masku.shape[1])
    # imm_noise = A * U
    c_imm_noise = np.zeros(cropped_img.shape, dtype = np.float)
    for kk in range(3):
        c_imm_noise[:,:,kk] = cimm[:,:,kk] + (A*np.random.rand(masku.shape[0],masku.shape[1]))
    return c_imm_noise


def mask_resize(img):
    width, height = img.size
    imgx = img.resize((cfg.WIDTH,cfg.WIDTH), Image.ANTIALIAS)
    return np.array(imgx, dtype=np.float32)

""" This has been changed to return a uint8 image as all images are normalized
    just before getting into the final list
"""
def img_tranfrom_8_points(img, bbox):
    width, height = img.size
    polygon = [(bbox[i]*width,bbox[i+1]*height) for i in range(0,len(bbox),2)]
    maskIm = Image.new('L', (width, height), 0)
    ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
    mask = np.array(maskIm)
    single_person = np.multiply(np.array(img),mask[:,:, np.newaxis])
    return single_person.astype(np.uint8), width, height

def img_tranfrom(img, bbox):
    width, height = img.size
    T = [width, height, width, height]
    mask = np.zeros((height,width))
    bbox= np.array(bbox)
    xmin, ymin, xmax, ymax = (bbox*T)
    mask[int(round(ymin)): int(round(ymax)), int(round(xmin)) : int(round(xmax))] = 1
    single_person = np.multiply(np.array(img),mask[:,:, np.newaxis])
    return single_person.astype(int), width, height

def read_image(img_path):
    return Image.open(img_path)

def pickle_writer(file_path, data):
    with open(file_path, "wb") as fp:
        pickle.dump(data, fp)

def pickle_reader(file_path):
    with open(file_path, "rb") as fp:
        return pickle.load(fp)
    


def min_max(image):
    mmin = np.min(image)
    mmax = np.max(image)
    return mmin, mmax

def normaliseTo01(image):
    mmin, mmax = min_max(image)
    if mmin != 0 or mmax != 1:
       new_min = image - mmin 
       newmax = new_min/np.max(new_min)
    else:
        newmax = image
    return newmax


def normaliseTouint8(image):
    mmin, mmax = min_max(image)
    if mmin>=0.0 and mmax<=1.0:
         new_imm = img_as_ubyte(image)
    else: 
         new_imm = img_as_ubyte(normaliseTo01(image))
    return new_imm