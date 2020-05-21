from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import to_categorical

import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import mtcnn
import numpy as np
import os
import random
import tensorflow as tf

# Fix progress bar
from tqdm.keras import TqdmCallback

# Imbalance weights
from sklearn.utils.class_weight import compute_sample_weight

from utils import Params


def generate_age_portion(i, age_portion):
    age = 0
    for index, portion in enumerate(age_portion):
        if portion[0] <= i <= portion[1]:
            age = index
    return age


def get_aug():
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    augmenter = iaa.Sequential(
        [
            sometimes(iaa.OneOf(
                [
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                    iaa.Affine(
                        scale={"x": (0.8, 1), "y": (0.8, 1)},
                        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                        rotate=(-10, 10),
                        shear=(-5, 5),
                        order=[0, 1],
                        cval=(0, 255),
                        mode='constant'

                    ),
                    iaa.PerspectiveTransform(scale=(0.01, 0.03)),
                    iaa.Crop(px=(4, 6)),
                    iaa.SaltAndPepper(p=(0.01,0.02)),
                    iaa.Fliplr(1.0)
                ]
            )),
        ],
        random_order=True)

    return augmenter


def build_model(ethics, ages, img_shape):
    backbone = InceptionV3(include_top=False, weights='imagenet', input_shape=img_shape)
    out_backbone = backbone.get_layer('mixed10').output

    # freeze pretrained weights for some epochs
    # for layer in backbone.layers:
    #     layer.trainable = False

    head1 = GlobalAveragePooling2D()(out_backbone)
    head1 = Flatten()(head1)
    out1 = Dense(2, activation='softmax')(head1)

    head2 = GlobalAveragePooling2D()(out_backbone)
    head2 = Dense(64, activation='relu')(head2)
    head2 = Dropout(0.2)(head2)
    out2 = Dense(len(ages), activation='softmax')(head2)

    head3 = GlobalAveragePooling2D()(out_backbone)
    head3 = Dense(64, activation='relu')(head3)
    head3 = Dropout(0.2)(head3)
    out3 = Dense(len(ethics), activation='softmax')(head3)

    age_loss = "categorical_crossentropy"
    gender_loss = "categorical_crossentropy"
    race_loss = "categorical_crossentropy"

    model = Model(backbone.input, [out1, out2, out3])
    model.summary()

    opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss = [gender_loss, age_loss, race_loss], optimizer=opt, metrics=["accuracy"])

    return model


def get_callbacks():
    filepath1 ="./weights/weights-race-{epoch:02d}-{val_dense_4_accuracy:.2f}.h5"
    checkpoint1 = ModelCheckpoint(filepath1, monitor='val_dense_4_accuracy', verbose=1, mode='auto', save_best_only=True)

    filepath2 ="./weights/weights-age-{epoch:02d}-{val_dense_2_accuracy:.2f}.h5"
    checkpoint2 = ModelCheckpoint(filepath2, monitor='val_dense_2_accuracy', verbose=1, mode='auto', save_best_only=True)

    def lr_scheduler(epoch, lr):
        decay_rate = 0.1
        decay_step = 50
        if epoch % decay_step == 0 and epoch > 10:
            lr = decay_rate*lr
        return lr

    scheduler = LearningRateScheduler(lr_scheduler, verbose=1)

    es = EarlyStopping(monitor='val_loss', patience=40, verbose=1, min_delta=1e-2)

    callbacks_list = [checkpoint2,  scheduler]

    return callbacks_list


def isotropically_resize_image(img, size, resample=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if w > h:
        h = h * size // w
        w = size
    else:
        w = w * size // h
        h = size

    resized = cv2.resize(img, (w, h), interpolation=resample)
    return resized


def make_square_image(img):
    h, w = img.shape[:2]
    size = max(h, w)
    t = 0
    b = size - h
    l = 0
    r = size - w
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)


def data_generator(lines, age_portion, ethics,  img_shape, aug=None, bs=1):
    data_list = list()
    for line in lines[1:]:
        data_list.append(line.rstrip())

    n_batches = int(len(data_list) / bs)

    while True:
        for i in range(n_batches):
            img_batch = []
            age_batch = []
            gender_batch = []
            race_batch = []

            x = data_list[i * bs:(i + 1) * bs]

            for index, j in enumerate(x):
                splitter = j.split(',')
                file_name = splitter[0]
                age = int(splitter[1])
                gender = int(splitter[2])
                race = int(splitter[3])

                if age >= 100:
                    age = 100

                img = cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB)

                if aug is not None:
                    img = aug.augment_image(img)

                # Resize to fixed size
                img = isotropically_resize_image(img, img_shape[0])
                img = make_square_image(img)

                img = preprocess_input(img)

                img_batch.append(img)
                age_batch.append(generate_age_portion(age, age_portion))
                gender_batch.append(gender)
                race_batch.append(race)

            images = np.reshape(img_batch, (bs, img_shape[0], img_shape[1], img_shape[2]))
            ages = to_categorical(age_batch, num_classes=len(age_portion))
            genders = to_categorical(gender_batch, num_classes=2)
            races = to_categorical(race_batch, num_classes=len(ethics))

            # Calculate sample weights
            age_weights = compute_sample_weight(class_weight='balanced',y=ages)
            gender_weights = compute_sample_weight(class_weight='balanced', y=genders)
            race_weight = compute_sample_weight(class_weight='balanced', y=races)

            yield images, [genders, ages, races], [age_weights, gender_weights, race_weight]


if __name__ == '__main__':
    params = Params('./params.json')

    image_shape = (params.img_width, params.img_height, params.img_depth)

    aug = get_aug()

    f = open('./train_utk.csv','r')
    train_list = f.readlines()
    f.close()

    f = open('./val_utk.csv','r')
    val_list = f.readlines()
    f.close()

    train_generator = data_generator(train_list, params.age_portion, params.ethics, image_shape, aug, bs=params.batch_size)
    val_generator = data_generator(val_list, params.age_portion, params.ethics, image_shape, None, bs=params.batch_size)

    model = build_model(params.ethics, params.age_portion, image_shape)
    model.fit(train_generator, steps_per_epoch=len(train_list)//params.batch_size,
                    validation_data=val_generator,
                    validation_steps=len(val_list)//params.batch_size,
                    epochs=100,
                    verbose=0,
                    callbacks=get_callbacks())

    model.save('./weights/weights_balance.h5')