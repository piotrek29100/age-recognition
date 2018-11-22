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
import cv2
from PIL import Image
import matplotlib.pyplot as plt

model_folder = "model"
train_dir = './imdb/train'
validation_dir = './imdb/test'


SAMPLES_RANGE = (0, 1000)
SAMPLES_NUMBER = SAMPLES_RANGE[1] - SAMPLES_RANGE[0]
VALIDATION_DATA = 0.2
TRAIN_SAMPLES = int(SAMPLES_NUMBER * (1-VALIDATION_DATA))
TEST_SAMPLES = int(SAMPLES_NUMBER * VALIDATION_DATA)

BATCH_SIZE = 10
STEPS_PER_EPOCH = int(TRAIN_SAMPLES/BATCH_SIZE)
VALIDATION_STEPS = int(TEST_SAMPLES/BATCH_SIZE)
EPOCHS = 1

def prepere_Xy_sets(folder_name="imdb/train", image_target_size=(224, 224), age_range=(10,80)):
    f = [[dirpath, dirnames, filenames] for dirpath, dirnames, filenames in os.walk(folder_name)]
    f = [[dirpath.split("\\")[-1], [cv2.imread(dirpath+"\\"+image_name) for image_name in filenames]] for dirpath, dirnames, filenames in f]
    f = [[age, image] for age, images in f for image in images if int(age)>=age_range[0] and int(age)<= age_range[1]]
    X = np.array([cv2.resize(image, dsize=image_target_size, interpolation=cv2.INTER_CUBIC) for age, image in f])
    y = np.array([(int(age)-age_range[0]) / (age_range[1]-age_range[0]) for age, image in f])

    return X, y

def decode_age(y, age_range=(10,80)):
    return (age_range[1]-age_range[0]) * y + 10

# image_array.shape = [height, width, 3]
def predict_age(image_array):
    y = model.predict(np.reshape(image_array, (1,image_array.shape[0],image_array.shape[1],image_array.shape[2])))
    return decode_age(y)[0][0]

def prepere_generators(samples_range):

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

    # train_generator = datagen.flow_from_directory(
    #     train_dir,
    #     target_size=(224, 224),
    #     batch_size=BATCH_SIZE,
    #     shuffle=True)

    train_X, train_y = prepere_Xy_sets("imdb/train")
    train_generator = datagen.flow(train_X[samples_range[0]:samples_range[1]], train_y[samples_range[0]:samples_range[1]],
        batch_size=BATCH_SIZE,
        shuffle=True)

    # this is a similar generator, for validation data
    # validation_generator = datagen.flow_from_directory(
    #     validation_dir,
    #     target_size=(224, 224),
    #     batch_size=BATCH_SIZE,
    #     class_mode='categorical')
    
    val_X, val_y = prepere_Xy_sets("imdb/test")
    validation_generator = datagen.flow(val_X[samples_range[0]:samples_range[1]], val_y[samples_range[0]:samples_range[1]],
        batch_size=BATCH_SIZE,
        shuffle=False)

    return train_generator, validation_generator, train_X, train_y, val_X, val_y

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
    base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(1, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    save_model(model, 'base_model.h5')
    return model

train_generator, validation_generator, train_X, train_y, val_X, val_y = prepere_generators(SAMPLES_RANGE)

# # Show image form image_generator
# x,y = validation_generator.next()
# for i in range(0,5):
#     image = 256 - x[i] * 255
#     image_copy = image.copy()
#     image[:,:,0] = image_copy[:,:,2]
#     image[:,:,1] = image_copy[:,:,1]
#     image[:,:,2] = image_copy[:,:,0]
#     plt.imshow(image)
#     plt.show()

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
model.compile(optimizer='adam', loss='mean_squared_error')

# train the model on the new data for a few epochs
model.fit_generator(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_steps=VALIDATION_STEPS,
                    epochs=EPOCHS)

save_model(model, "model_1.h5")

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
              loss='mean_squared_error')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_steps=VALIDATION_STEPS,
                    epochs=EPOCHS)

save_model(model, "model_1.h5")

# Check results
y = predict_age(train_X[0])

# Validation set
p = model.predict_generator(validation_generator)
y_pred = decode_age(p).flatten()
y_val = decode_age(val_y)
y_pred
y_val

# Train set
p = model.predict_generator(train_generator)
y_pred = decode_age(p).flatten()
y_train = decode_age(train_y)
y_pred
y_train