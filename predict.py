"""Define the model."""
import os
import tensorflow as tf
import cv2
import numpy as np
import glob
from PIL import ImageFont, ImageDraw, Image
import argparse


# Comment out waiting for Tensorflow 2.0 doc
# from tensorflow import keras as K
import mtcnn

from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
# Augmentation
import random
import imgaug as ia
import imgaug.augmenters as iaa

import random
from utils import isotropically_resize_image, make_square_image, Params


def convert_age(age, age_portion):
    portion = age_portion[age]
    result = (portion[0], portion[1])
    return result

def face_detection(face_detector, image):
    result = list()
    # Image is in RGB for mtcnn

    faces = detector.detect_faces(image)
    sorted(faces, key=lambda x: x['box'][2]*x['box'][3], reverse=False)

    h,w,_ = image.shape
    for face in faces:
        if face['confidence'] > 0.9 and len(face['keypoints']) >= 4:
            box = face['box']

            # Get face width and height
            width = box[3] + box[1]
            height = box[2] + box[0]

            # Crop with margin
            mar_x = width * 0.1
            mar_y = height * 0.1

            xmin = box[1] - int(mar_x/2)
            if xmin < 0:
                xmin = box[1]

            xmax = box[3] + box[1] + int(mar_x/2)
            if xmax > w:
                xmax = box[3]+box[1]

            ymin = box[0] - int(mar_y/2)
            if ymin < 0:
                ymin = box[0]

            ymax = box[2] + box[0] + int(mar_y/2)
            if ymax > h:
                ymax = box[2]+box[0]

            face_img = image[xmin:xmax,ymin:ymax]

            result.append(face_img)

    return result


def predict_multitask(model, face_list, img_shape, genders, age_portion, ethics):
    result = list()
    if len(faces) == 0:
        return result

    # Sort the face with descending order of area
    for face in faces:
        request_face = isotropically_resize_image(face, img_shape[0])
        request_face = make_square_image(request_face)

        request_face = preprocess_input(request_face)

        pred = model.predict(np.reshape(request_face, (1, img_shape[1], img_shape[0], img_shape[2])))

        g = genders[np.argmax(pred[0])]
        a = convert_age(np.argmax(pred[1]), age_portion)
        e = ethics[np.argmax(pred[2])]

        result.append((g,a,e))

    return result


parser = argparse.ArgumentParser()
parser.add_argument('--model_weights', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    detector = mtcnn.MTCNN()
    model = load_model(args.model_weights)

    params = Params('./params.json')

    image_shape = (params.img_width, params.img_height, params.img_depth)

    # Detect face
    test_image = cv2.imread('./sample_img/female_white_67.jpg')
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

    faces = face_detection(detector, test_image)
    preds = predict_multitask(model, faces, image_shape, params.gender, params.age_portion, params.ethics)

    print(preds)

    test_image = cv2.imread('./sample_img/male_asian_24.jpg')
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

    faces = face_detection(detector, test_image)
    preds = predict_multitask(model, faces, image_shape, params.gender, params.age_portion, params.ethics)

    print(preds)

    test_image = cv2.imread('./sample_img/female_indian_23.jpg')
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

    faces = face_detection(detector, test_image)
    preds = predict_multitask(model, faces, image_shape, params.gender, params.age_portion, params.ethics)

    print(preds)

    test_image = cv2.imread('./sample_img/female_black_24.jpg')
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

    faces = face_detection(detector, test_image)
    preds = predict_multitask(model, faces, image_shape, params.gender, params.age_portion, params.ethics)

    print(preds)