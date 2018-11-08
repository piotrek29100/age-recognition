import pandas as pd
import numpy as np
import face_recognition
from PIL import Image
import os

df = pd.read_csv("imdb_crop\\imdb.csv")

target_directory = 'imdb\\test'

if not os.path.exists(target_directory):
    os.makedirs(target_directory)


def crop_image(image):
    top, right, bottom, left = face_recognition.face_locations(image)[0]

    image = image[top:bottom, left:right]
    image = Image.fromarray(image)
    return image

for [age, path, x1, y1, x2, y2, face_score, second_face_score] in df[1000:1200].values:

    if age < 20 or age > 40:
        continue

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