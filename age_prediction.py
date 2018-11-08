from PIL import Image
import face_recognition
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

model_folder = "model"
model = load_model("{}/model_1.h5".format(model_folder))

datagen = ImageDataGenerator(rescale=1./255)

def crop_face(image):
    top, right, bottom, left = face_recognition.face_locations(image)[0]
    #top, right, bottom, left = face_recognition.face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")[0]

    image = image[top:bottom, left:right]
    image = Image.fromarray(image)
    return image

def predict_age(data):
    data = np.asarray(img, dtype="float32")

    # reshape to be [samples][width][height][pixels]
    X_test = data.reshape(1, data.shape[0], data.shape[1], 3)
    # convert from int to float
    X_test = X_test.astype('float32')

    p = model.predict_generator(datagen.flow(X_test), verbose=0)

    return np.argmax(p) + 20

img = Image.open("./imdb/test/26/nm4141252_rm965800704_1989-4-5_2015.jpg")
img.load()

predict_age(img)


