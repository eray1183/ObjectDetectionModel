from google.colab import drive
drive.mount('/content/gdrive')

ROOT_DIR ='/content/gdrive/My Drive/eray test/'

import tensorflow as tf
import yaml
import os
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
from keras.layers import MaxPooling2D
import matplotlib.pyplot as plt


def load_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def process_data(image_dir, label_dir, input_shape):
    images = []
    labels = []

    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + '.txt')

            img = load_img(img_path, target_size=input_shape[:2])
            img = img_to_array(img) / 255.0

            with open(label_path, 'r') as file:
                for line in file:
                    class_id, xmin, ymin, xmax, ymax = map(float, line.strip().split())
                    bbox = [xmin / input_shape[1], ymin / input_shape[0], xmax / input_shape[1], ymax / input_shape[0], class_id]
                    images.append(img)
                    labels.append(bbox)

    return np.array(images), np.array(labels)

yaml_path = '/content/gdrive/My Drive/ainos_v1_config2.yaml'

yaml_data = load_yaml(yaml_path)
train_image_dir = yaml_data['train']
val_image_dir = yaml_data['val']
train_label_dir = yaml_data['train_labels']
val_label_dir = yaml_data['val_labels']

input_shape = (416, 416, 3)

x_train, y_train = process_data(train_image_dir, train_label_dir, input_shape)
x_test, y_test = process_data(val_image_dir, val_label_dir, input_shape)

from tensorflow.keras.layers import BatchNormalization, Dropout
def improved_yolo(input_shape, grid_size, num_classes):
    inputs = Input(shape=input_shape)

    # İlk Konvolüsyon Bloğu
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # İkinci Konvolüsyon Bloğu
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # Üçüncü Konvolüsyon Bloğu
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # Dördüncü Konvolüsyon Bloğu
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # Düzleştirme ve Yoğun Katmanlar
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)  # Dropout oranını azalttık
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)  # Dropout oranını azalttık
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)

    # Çıkış Katmanı
    num_boxes = 1
    num_bbox_attributes = 5
    outputs = Dense(num_boxes * num_bbox_attributes, activation='sigmoid')(x)

    # Modeli Oluşturma
    model = Model(inputs, outputs)
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

    return model

metrics = ['accuracy', MeanSquaredError(), MeanAbsoluteError()]

grid_size = 7
num_classes = len(yaml_data['names'])
model = improved_yolo(input_shape, grid_size, num_classes)

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=metrics)

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

model.summary()

history = model.fit(x_train, y_train, epochs=50, batch_size=16,
                    validation_data=(x_test, y_test), callbacks=[early_stopping])

print("Accuracy:", history.history['accuracy'][-1])
print("MSE:", history.history['mean_squared_error'][-1])
print("MAE:", history.history['mean_absolute_error'][-1])

#Nesne tanıma yapabilmek için CNN kullanarak model oluşturdum fakat accurary değeri 0.55 üzerine çıkamadığı için predict yapmadım.
