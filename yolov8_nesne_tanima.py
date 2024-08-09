from ultralytics import YOLO
import cv2 as cv
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np

import os

model = YOLO("yolov8m.yaml")

results=model.train(data=os.path.join("ainos_v1_config.yaml"), epochs=50)

"""
.yaml dosyası
//path: '/content/gdrive/My Drive'  # dataset root dir
train: '/content/gdrive/My Drive/ainos_model_v1_photo&label/images/train'  # train images (relative to 'path')
val: '/content/gdrive/My Drive/ainos_model_v1_photo&label/images/test'  # val images (relative to 'path')
train_labels: '/content/gdrive/My Drive/ainos_model_v1_photo&label/labels/train'  # train labels (relative to 'path')
val_labels: '/content/gdrive/My Drive/ainos_model_v1_photo&label/labels/test'  # val labels (relative to 'path')

names:
  0: 'mouse'
  1: 'papia'
  2: 'headphone'
"""
