import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf

from resnet import ResnetParam, ResnetModel

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train[...,np.newaxis].astype(np.float32)
x_test = x_test[...,np.newaxis].astype(np.float32)

y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

x_train /= 255
x_test /= 255

# datagen = tf.keras.preprocessing.image.ImageDataGenerator()
# train_data = datagen.flow(x_train, y_train, batch_size=32)
# validation_data = datagen.flow(x_train, y_train, batch_size=32)

param = ResnetParam()
param.input_shape = [32, 28, 28, 1]
param.num_classes = 10
param.num_layers = 18
param.learning_rate = 1e-4
param.logdir = './train_log'
param.num_epochs = 100

model = ResnetModel(param)
model.train((x_train, y_train), (x_test, y_test))

print()