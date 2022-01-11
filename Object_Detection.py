# -*- coding: utf-8 -*-
#                                                                             
# PROGRAMMER: Karim Elshetihy
# DATE CREATED: 27/12/2021                     
#
##

# Importing Packages
import numpy as np
import cv2
from get_args import get_args
from predict import get_objects

readVideo = False

# Importing Arguments from 'get_args.py'
arguments = get_args()

# Initialize list of class labels that MobileNet_SSD was trained on, 
# then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


# load our serialized model from disk
print("### Loading Model...")
net = cv2.dnn.readNetFromCaffe(arguments["prototxt"], arguments["model"])

# Loading the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it via the 
# authors of the MobileNet SSD implementation

if arguments["image"].endswith(".mp4"):
    video = cv2.VideoCapture(arguments["image"])
    readVideo = True

    while readVideo:
        ret, image = video.read()
        get_objects(arguments, net, image, CLASSES, COLORS)        

        # Break with ESC
        if cv2.waitKey(1) >= 0:
            break

else:
    image = cv2.imread(arguments["image"])
    (height, width) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    get_objects(arguments, net, image, CLASSES, COLORS) 
    cv2.waitKey(0)