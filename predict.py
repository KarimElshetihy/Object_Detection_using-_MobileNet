import numpy as np
import cv2
from get_args import get_args

# Importing Arguments from 'get_args.py'
arguments = get_args()

def get_objects(arguments, net, image, CLASSES, COLORS):
    (height, width) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # Pass the blob through the network and obtain the detections and class prediction
    print("### Detecting Objects...")
    net.setInput(blob)
    detections = net.forward()

    # Looping over the detections
    for object in np.arange(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, object, 2]
        
        # Filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        if confidence > arguments["confidence"]:
            
            # Extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for the object
            idx = int(detections[0, 0, object, 1])
            box = detections[0, 0, object, 3:7] * np.array([width, height, width, height])
            
            (startX, startY, endX, endY) = box.astype("int")

            # Display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("### {}".format(label))
            
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
            
            y = (startY - 20) if ((startY - 20) > 20) else (startY + 20)
            
            cv2.putText(image, label, (startX+5, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[idx], 1)


    # Show the output image
    cv2.imshow("Output", image)
