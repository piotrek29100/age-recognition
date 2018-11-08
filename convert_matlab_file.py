import scipy.io as scipy_io
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

mat = scipy_io.loadmat('imdb_crop/imdb.mat')
images_data = mat["imdb"]

images_number = images_data[0,0][0].shape[1]

def convert_to_date(matlab_datenum):
    if (matlab_datenum < 365*1800):
        return datetime.fromordinal(1)
    return datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum%1) - timedelta(days = 366)

def parse(images_data, i):
    matlab_datenum = int(images_data[0,0]['dob'][0,i])
    data_of_birth = convert_to_date(matlab_datenum)
    
    photo_taken = int(images_data[0,0]['photo_taken'][0,i])
    full_path = str(images_data[0,0]['full_path'][0,i])

    face_location = images_data[0,0]['face_location'][0,i]
    face_score = float(images_data[0,0]['face_score'][0,i])
    second_face_score = float(images_data[0,0]['second_face_score'][0,i])

    age = photo_taken - data_of_birth.year

    return [age, full_path.split("'")[1] , face_location[0,0], face_location[0,1], face_location[0,2], face_location[0,3], face_score, second_face_score]

images = [parse(images_data, i) for i in range(0, images_number-1)]

df = pd.DataFrame(columns=['age', 'path', 'x1', 'y1', 'x2', 'y2', 'face_score', 'second_face_score'], data=images)

df = df[np.isnan(df["second_face_score"])]
df = df[~np.isnan(df["face_score"])]
df = df[~np.isinf(df["face_score"])]

df = df.sort_values(['face_score'], ascending=False)

df.to_csv("imdb_crop\imdb.csv", index=False)

