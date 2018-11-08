import pandas as pd
import numpy as np
import face_recognition
from PIL import Image
import os
import random

def crop_image(image):
    top, right, bottom, left = face_recognition.face_locations(image)[0]

    image = image[top:bottom, left:right]
    image = Image.fromarray(image)
    return image

def generate_images(source_folder_name, images_number, validation_data=0.2, min_age=10, max_age=80):

    # Zdjęcia sa postortowane od tych na których wyraźnie widać twarz do tych najmniej wyraźnych
    df = pd.read_csv("{}\\imdb.csv".format(source_folder_name))

    df = df[~(df.age<min_age) & ~(df.age>max_age)]

    order = random.sample(range(0,images_number),images_number)
    validation_split = int(images_number*(1-validation_data))
    train_values = df.iloc[order[:validation_split]].values
    test_values = df.iloc[order[validation_split:]].values

    _generate_images('imdb\\train', train_values)
    _generate_images('imdb\\test', test_values)


def _generate_images(target_directory, values):

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    for [age, path, x1, y1, x2, y2, face_score, second_face_score] in values:

        path = path.replace('/', '\\')
        full_path = 'imdb_crop\\{}'.format(path)

        image = face_recognition.load_image_file(full_path)
        #  x1, y1, x2, y2 coordinates from imdb.mat are broken
        image = crop_image(image)
        #image.show()

        directory = '{}\\{}'.format(target_directory,age)
        new_file = path.split('\\')[1]

        if not os.path.exists(directory):
            os.makedirs(directory)

        image.save('{}\\{}'.format(directory, new_file))

# 184'623 images
generate_images("imdb_crop", 50000, validation_data=0.2, min_age=10, max_age=80)