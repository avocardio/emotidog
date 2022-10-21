
# Showcase each image from folder "dog" and have it press 1 to classify as 
# happy and move to folder "happy", 2 to classify as alert and move to folder "alert", 
# and 3 to classify as content and move to folder "content", and 4 to classify as
# curious and move to folder "curious", and 5 to classify as sad and move to folder "sad", 
# and 6 to classify as angry and move to folder "angry". 

# If you press any other key, the image will be moved to folder "unknown".

import cv2
import os
import shutil
import numpy as np
from PIL import Image
import time
import sys


def label_dogs(folder):
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(folder, filename))
            cv2.imshow('image', img)
            k = cv2.waitKey(0)
            if k == ord('1'):
                shutil.move(os.path.join(folder, filename), os.path.join("happy", filename))
            elif k == ord('2'):
                shutil.move(os.path.join(folder, filename), os.path.join("alert", filename))
            elif k == ord('3'):
                shutil.move(os.path.join(folder, filename), os.path.join("content", filename))
            elif k == ord('4'):
                shutil.move(os.path.join(folder, filename), os.path.join("curious", filename))
            elif k == ord('5'):
                shutil.move(os.path.join(folder, filename), os.path.join("sad", filename))
            elif k == ord('6'):
                shutil.move(os.path.join(folder, filename), os.path.join("angry", filename))
            else:
                shutil.move(os.path.join(folder, filename), os.path.join("unknown", filename))
            cv2.destroyAllWindows()


label_dogs("dog")