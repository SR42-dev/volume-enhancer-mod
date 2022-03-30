import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf

import cv2
import os

import numpy as np

import librosa
import matplotlib.pyplot as plt
import librosa.display

import os

import pickle

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def waveletGen(cls):
    img_names = os.listdir('genres/' + cls)
    os.makedirs('wavelets/train/' + cls)
    os.makedirs('wavelets/test/' + cls)
    print(cls)
    train_names = img_names[:60]
    test_names = img_names[60:]
    cnt = 0
    for nm in train_names:
        cnt += 1
        x, sr = librosa.load('genres/' + cls + '/' + nm)
        # plt.figure(figsize=(14, 5))
        librosa.display.waveplot(x)
        plt.savefig('wavelets/train/' + cls + '/' + str(cnt) + '.png')
        plt.close()

    cnt = 0
    for nm in test_names:
        cnt += 1
        x, sr = librosa.load('genres/' + cls + '/' + nm)
        # plt.figure(figsize=(14, 5))
        librosa.display.waveplot(x)
        plt.savefig('wavelets/test/' + cls + '/' + str(cnt) + '.png')
        plt.close()


def spectrogramGen(cls):
    img_names = os.listdir('genres/' + cls)
    os.makedirs('spectrogram/train/' + cls)
    os.makedirs('spectrogram/test/' + cls)
    print(cls)
    train_names = img_names[:60]
    test_names = img_names[60:]
    cnt = 0
    for nm in train_names:
        cnt += 1
        x, sr = librosa.load('genres/' + cls + '/' + nm)
        X = librosa.stft(x)
        Xdb = librosa.amplitude_to_db(abs(X))
        librosa.display.specshow(Xdb)
        plt.savefig('spectrogram/train/' + cls + '/' + str(cnt) + '.png')
        plt.close()

    cnt = 0
    for nm in test_names:
        cnt += 1
        x, sr = librosa.load('genres/' + cls + '/' + nm)
        X = librosa.stft(x)
        Xdb = librosa.amplitude_to_db(abs(X))
        librosa.display.specshow(Xdb)
        plt.savefig('spectrogram/test/' + cls + '/' + str(cnt) + '.png')
        plt.close()




