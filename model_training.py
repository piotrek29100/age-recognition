# categorical >> float ??
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# https://keras.io/applications/
# https://keras.io/preprocessing/image/

import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.applications import vgg16, inception_v3, resnet50, mobilenet
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import os

model_folder = "model"
train_dir = './imdb/train'
validation_dir = './imdb/test'

SAMPLES_NUMBER = 500
VALIDATION_DATA = 0.2
TRAIN_SAMPLES = int(SAMPLES_NUMBER * (1-VALIDATION_DATA))
TEST_SAMPLES = int(SAMPLES_NUMBER * VALIDATION_DATA)

BATCH_SIZE = 10
STEPS_PER_EPOCH = int(TRAIN_SAMPLES/BATCH_SIZE)
VALIDATION_STEPS = int(TEST_SAMPLES/BATCH_SIZE)
EPOCHS = 1

CLASSES_NUMBER = 71 # ages = <10, 80>

def prepere_generators():

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

    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True)

    # this is a similar generator, for validation data
    validation_generator = datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    return train_generator, validation_generator

def save_model(model, model_name):

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    model.save("{}/{}.h5".format(model_folder, model_name))

def create_model():

    # #Load the VGG model
    # # vgg_model = vgg16.VGG16(weights='imagenet')

    # #Load the ResNet50 model
    # resnet_model = resnet50.ResNet50(weights='imagenet')

    # #Load the MobileNet model
    # mobilenet_model = mobilenet.MobileNet(weights='imagenet')

    # #Load the Inception_V3 model
    # inception_model = inception_v3.InceptionV3(weights='imagenet',
    #                   include_top=False,
    #                   input_shape=(224, 224, 3))

    # create the base pre-trained model
    base_model = inception_v3.InceptionV3(
        weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have ... classes
    predictions = Dense(CLASSES_NUMBER, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    save_model(model, 'base_model.h5')
    return model

train_generator, validation_generator = prepere_generators()

model = create_model()
# OR
model = load_model("{}/model_1.h5".format(model_folder))

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(model.layers):
    print(i, layer.name)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in model.layers[:311]:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new data for a few epochs
model.fit_generator(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_steps=VALIDATION_STEPS,
                    epochs=EPOCHS)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
frozen_layers = 249
for layer in model.layers[:frozen_layers]:
    layer.trainable = False
for layer in model.layers[frozen_layers:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_steps=VALIDATION_STEPS,
                    epochs=EPOCHS)

save_model(model, "model_1.h5")

# Tests

#type(validation_generator)
#validation_generator.classes

#p = model.predict_generator(validation_generator)
#y = [np.argmax(x) for x in p]

#sum(pow(abs(y - validation_generator.classes), 2))/len(y)
