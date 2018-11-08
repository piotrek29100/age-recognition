# categorical >> float
    # https://keras.io/preprocessing/image/
    # tutaj chyba lepiej będzie zastosować sparse, a najlepiej własną funkcję mapującą wiek na 0-1

# Most important source: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# Prepare Generator 
train_dir = './imdb/train'
validation_dir = './imdb/test'

datagen = ImageDataGenerator(rescale=1./255)

# datagen = ImageDataGenerator(
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest')

batch_size = 20

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

# this is a similar generator, for validation data
validation_generator = datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

# From https://keras.io/applications/

from keras.applications import vgg16, inception_v3, resnet50, mobilenet
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

def create_model():
    

# #Load the VGG model
# vgg_model = vgg16.VGG16(weights='imagenet')
 
# #Load the ResNet50 model
# resnet_model = resnet50.ResNet50(weights='imagenet')
 
# #Load the MobileNet model
# mobilenet_model = mobilenet.MobileNet(weights='imagenet')
 
# #Load the Inception_V3 model
# inception_model = inception_v3.InceptionV3(weights='imagenet',
#                   include_top=False,
#                   input_shape=(224, 224, 3))

# create the base pre-trained model
base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have ... classes
predictions = Dense(21, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new data for a few epochs
model.fit_generator(train_generator,
        validation_data=validation_generator,
        samples_per_epoch=50,
        nb_epoch=10)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(train_generator,
        validation_data=validation_generator,
        samples_per_epoch=50,
        nb_epoch=10)

# Save model

import os
model_folder = "model"

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

model.save("{}/model_1.h5".format(model_folder))

# Validation

validation_generator = datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

type(validation_generator)
validation_generator.classes

p = model.predict_generator(validation_generator)
y = [np.argmax(x) for x in p]

sum(pow(abs(y - validation_generator.classes),2))/len(y)