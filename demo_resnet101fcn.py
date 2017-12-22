
# coding: utf-8

# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.

# In[1]:

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import LIP
import utils
import model as modellib
import visualize
import scipy.io
import h5py
#import seaborn
# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
# Download this file and place in the root of your 
# project (See README file for details)
LIP_MODEL_PATH = '/home/ltp/WorkShop/Human_Parsing/Resnet101_FCN/logs/Resnet101PFN/Titan/lip20171219T1805/Resnet101FCN_lip_0047.h5'
# Directory of images to run detection on
#IMAGE_DIR =os.path.join(ROOT_DIR, "images")
IMAGE_DIR= '/home/ltp/图片/smplayer_screenshots/crop'
train_dir ='/media/ltp/40BC89ECBC89DD32/LIPHP_data/LIP/SinglePerson/LIP_dataset/train_set/images/'

# ## Configurations
# 
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
# 
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.

# In[2]:

class InferenceConfig(LIP.LIPConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MAX_DIM=640
config = InferenceConfig()
config.display()


# ## Create Model and Load Trained Weights

# In[3]:

# Create model object in inference mode.
model = modellib.Resnet101FCN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(LIP_MODEL_PATH, by_name=True)


# ## Class Names
# 
# The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
# 
# To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
# 
# To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
# ```
# # Load COCO dataset
# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()
# 
# # Print class names
# print(dataset.class_names)
# ```
# 
# We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)

# In[4]:

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG','Hat', 'Hair','Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coats', 'Socks', 'Pants','Jumpsuits',  'Scarf', 'Skirt',  'Face', 'Left-arm','Right-arm',   'Left-leg','Right-leg','Left-shoe','Right-shoe' ]



# ## Run Object Detection

# In[5]:
#file_names = next(os.walk(IMAGE_DIR))[2]
#for v in file_names:
    #image = skimage.io.imread(os.path.join(IMAGE_DIR,v))
    
    ## Run detection
    #results = model.detect([image], verbose=0)
    
    ## Visualize results
    #r = results[0]
    
    #im=visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                #class_names, r['scores'])
    #skimage.io.imsave('/home/ltp/图片/smplayer_screenshots/crop/I2/'+v,im)
# Load a random image from the images folderim,ax =plt.subplots(1,figsize = (16,16))
IMAGE_DIR ='./images/'
file_names = '2001_430259.jpg'#next(os.walk(IMAGE_DIR))[2]
random.seed(120)
image = skimage.io.imread(IMAGE_DIR+file_names)
#seg=skimage.io.imread('/media/ltp/40BC89ECBC89DD32/LIPHP_data/LIP/SinglePerson/parsing/val_segmentations/2001_430259.png')
# Run detection
results = model.detect([image], verbose=0)
#results = model.detect_filter([image],f['label'][...].transpose(), verbose=0)
# Visualize results
r = results[0]
color = visualize.random_colors(N=config.NUM_CLASSES)
#im = visualize.apply_mask(image, seg, color=color, class_ids=[v for v in range(1,config.NUM_CLASSES)])
im = visualize.apply_mask(image, r['masks'], color=color, class_ids=[v for v in range(1,config.NUM_CLASSES)])


plt.imshow(im)
plt.show()






