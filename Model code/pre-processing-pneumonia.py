import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import cv2

TRAIN_PATH_1 = 'train/NORMAL/'
TRAIN_PATH_2 =  'train/PNEuMONIA/'
TEST_PATH_1 =  'test/NORMAL/'
TEST_PATH_2 =  'test/PNEUMONIA/'
VAL_PATH_1 =  'val/NORMAL/'
VAL_PATH_2 =  'val/PNEUMONIA/'

PP_PATH = os.path.join(os.getcwd(), '../')

    
def read_img(img_path):
    img = cv2.imread(img_path,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(128,128))
    return img

for img_path in tqdm(os.listdir(TRAIN_PATH_1)):
    cv2.imwrite(PP_PATH+TRAIN_PATH_1+img_path,read_img(TRAIN_PATH_1 + img_path))

for img_path in tqdm(os.listdir(TRAIN_PATH_2)):
    cv2.imwrite(PP_PATH+TRAIN_PATH_2+img_path,read_img(TRAIN_PATH_2 + img_path))

for img_path in tqdm(os.listdir(TEST_PATH_1)):
    cv2.imwrite(os.path.join(PP_PATH,TEST_PATH_1,img_path),read_img(TEST_PATH_1 + img_path))

for img_path in tqdm(os.listdir(TEST_PATH_2)):
    cv2.imwrite(os.path.join(PP_PATH,TEST_PATH_2,img_path),read_img(TEST_PATH_2 + img_path))
    

for img_path in tqdm(os.listdir(VAL_PATH_1)):
    cv2.imwrite(os.path.join(PP_PATH,VAL_PATH_1,img_path),read_img(VAL_PATH_1 + img_path))

for img_path in tqdm(os.listdir(VAL_PATH_2)):
    cv2.imwrite(os.path.join(PP_PATH,VAL_PATH_2,img_path),read_img(VAL_PATH_2 + img_path))
    
del img_path




# =============================================================================
# 
# =============================================================================

#JPEG TO PNG CONVERTER
import os
from PIL import Image
from tqdm import tqdm

target_directory1 = 'test/NORMAL/'
target_directory2 = 'test/PNEUMONIA/'
target_directory3 = 'train/NORMAL/'
target_directory4 = 'train/PNEUMONIA/'
target_directory5 = 'val/NORMAL/'
target_directory6 = 'val/PNEUMONIA/'

target = '.png'

for file in tqdm(os.listdir(target_directory1)):
    filename, extension = os.path.splitext(file)
    img = Image.open(target_directory1+filename + extension)
    img.save('NEW/'+target_directory1+filename + target)

for file in tqdm(os.listdir(target_directory2)):
    filename, extension = os.path.splitext(file)
    img = Image.open(target_directory2+filename + extension)
    img.save('NEW/'+target_directory2+filename + target)


for file in tqdm(os.listdir(target_directory3)):
    filename, extension = os.path.splitext(file)
    img = Image.open(target_directory3+filename + extension)
    img.save('NEW/'+target_directory3+filename + target)


for file in tqdm(os.listdir(target_directory4)):
    filename, extension = os.path.splitext(file)
    img = Image.open(target_directory4+filename + extension)
    img.save('NEW/'+target_directory4+filename + target)
    
for file in tqdm(os.listdir(target_directory5)):
    filename, extension = os.path.splitext(file)
    img = Image.open(target_directory5+filename + extension)
    img.save('NEW/'+target_directory5+filename + target)


for file in tqdm(os.listdir(target_directory6)):
    filename, extension = os.path.splitext(file)
    img = Image.open(target_directory6+filename + extension)
    img.save('NEW/'+target_directory6+filename + target)

