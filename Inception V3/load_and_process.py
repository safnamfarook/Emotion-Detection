import pandas as pd
import cv2
import numpy as np


imgs_path = 'fer2013/fer2013/fer2013.csv'
im_sz=(48,48)

def load_fer2013():
        imgs = pd.read_csv(imgs_path)
        px = imgs['pixels'].tolist()
        width, height = 48, 48
        face_dets = []
        for pixel_sequence in px:
            face_det = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face_det = np.asarray(face_det).reshape(width, height)
            face_det = cv2.resize(face_det.astype('uint8'),ima_sz)
            face_dets.append(face_det.astype('float32'))
        face_dets = np.asarray(face_dets)
        face_dets = np.expand_dims(face_dets, -1)
        emotions = pd.get_dummies(imgs['emotion']).values
        return face_dets, emotions

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x