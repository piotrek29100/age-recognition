"""
Face detection
"""
import cv2
import os
from time import sleep
import numpy as np
import argparse
from keras.utils.data_utils import get_file
from keras.models import load_model
from PIL import Image
import face_recognition
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

class FaceCV(object):
    """
    Singleton class for face recongnition task
    """

    def __init__(self, depth=16, width=8, face_size=64):
        self.face_size = face_size
        
        ## Modification
        model_folder = "model"
        self.model = load_model("{}/model_1.h5".format(model_folder))

    def predict_age(self, data):
        datagen = ImageDataGenerator(rescale=1./255)
        data = np.asarray(data, dtype="float32")

        # reshape to be [samples][width][height][pixels]
        X_test = data.reshape(1, data.shape[0], data.shape[1], 3)
        # convert from int to float
        X_test = X_test.astype('float32')

        p = self.model.predict_generator(datagen.flow(X_test), verbose=0)

        return np.argmax(p) + 10

    def crop_face(self, imgarray, section, margin=40, size=64):
        """
        :param imgarray: full image
        :param section: face detected area (x, y, w, h)
        :param margin: add some margin to the face detected area to include a full head
        :param size: the result image resolution with be (size x size)
        :return: resized image in numpy array with shape (size x size x 3)
        """
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w,h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w-1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h-1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

    def detect_face(self):
        face_cascade = cv2.CascadeClassifier(self.CASE_PATH)

        # 0 means the default video capture device in OS
        video_capture = cv2.VideoCapture(0)
        # infinite loop, break by key ESC
        while True:
            sleep(0.2)
            if not video_capture.isOpened():
                sleep(5)
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=10,
                minSize=(self.face_size, self.face_size)
            )
            # placeholder for cropped faces
            face_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))
            for i, face in enumerate(faces):
                face_img, cropped = self.crop_face(frame, face, margin=40, size=self.face_size)
                (x, y, w, h) = cropped
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                face_imgs[i,:,:,:] = face_img
            if len(face_imgs) > 0:
                # predict ages and genders of the detected faces
                result = self.model.predict(face_imgs)
            # draw results
            for i, face in enumerate(faces):
                label = "{}".format(result)
                self.draw_label(frame, (face[0], face[1]), label)

            cv2.imshow('Keras Faces', frame)
            if cv2.waitKey(5) == 27:  # ESC key press
                break
        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()

def main():
    face = FaceCV()

    face.detect_face()

if __name__ == "__main__":
    main()
