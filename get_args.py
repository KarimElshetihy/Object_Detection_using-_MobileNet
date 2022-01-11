# -*- coding: utf-8 -*-
#                                                                             
# PROGRAMMER: Karim Elshetihy
# DATE CREATED: 27/12/2021                     
#
#
# PURPOSE: Create a function that retrieves the needed command line inputs 
#          from the user using the Argparse Python module. If the user fails to 
#          provide some or all of the inputs, then the default values are
#          used for the missing inputs. 
# 
# Model: We use a pre-trained MobileNet taken from https://github.com/chuanqi305/MobileNet-SSD/ 
#        that was trained on the Caffe-SSD framework.
#     1. Prototxt file is the text network file for Caffe Model
#     2. Caffe file is the Caffe Model
#
# Command Line Arguments:
#     1. Image or Video as --image with default value 'Images/0.1.jpg'
#     2. Prototxt as --prototxt with default value 'Model/MobileNetSSD_deploy.prototxt'
#     3. Model as --model with default value 'Model/MobileNetSSD_deploy.caffe'
#     4. Confidence as --confidence with default value '0.4'
#
##

# Importing Packages
import argparse

# Define get_input_args function 
def get_args():
    """
         PURPOSE: Create a function that retrieves the needed command line inputs 
                  from the user using the Argparse Python module. If the user fails to 
                  provide some or all of the inputs, then the default values are
                  used for the missing inputs. 
         
         Model: We use a pre-trained MobileNet taken from https://github.com/chuanqi305/MobileNet-SSD/ 
                that was trained on the Caffe-SSD framework.
             1. Prototxt file is the text network file for Caffe Model
             2. Caffe file is the Caffe Model
        
         Command Line Arguments:
             1. Image as --image with default value 'Images/0.1.jpg'
             2. Prototxt as --prototxt with default value 'Model/MobileNetSSD_deploy.prototxt'
             3. Model as --model with default value 'Model/MobileNetSSD_deploy.caffe'
             4. Confidence as --confidence with default value '0.4'
    """
    # Building Argument Parser & Parse the Arguments
    argsParser = argparse.ArgumentParser()
    argsParser.add_argument("-i", "--image", default="Images/01.JPG", help="Path to the Input Image or Video")
    argsParser.add_argument("-p", "--prototxt", default="Model/MobileNetSSD_deploy.prototxt", help="Path to the Caffe Prototxt File")
    argsParser.add_argument("-m", "--model", default="Model/MobileNetSSD_deploy.caffemodel", help="Path to the Caffe Pre-trained Model")
    argsParser.add_argument("-c", "--confidence", type=float, default=0.4, help="Minimum Probability to Filter Detections")
    
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function
    arguments = vars(argsParser.parse_args())
    
    return arguments
