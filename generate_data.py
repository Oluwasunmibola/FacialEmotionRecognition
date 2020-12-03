import numpy as np
import pandas as pd
import os
from PIL import Image

# reading the csv file
df = pd.read_csv("C:/Users/sbola/PycharmProjects/FacialRecognition/FacialDetectionData/fer2013.csv")
df0 = df[df['emotion'] == 0]
df1 = df[df['emotion'] == 1]
df2 = df[df['emotion'] == 2]
df3 = df[df['emotion'] == 3]
df4 = df[df['emotion'] == 4]
df5 = df[df['emotion'] == 5]
df6 = df[df['emotion'] == 6]

# making various directories of facial expression
os.mkdir('C:/Users/sbola/PycharmProjects/FacialRecognition/FacialDetectionData/Angry/')
os.mkdir('C:/Users/sbola/PycharmProjects/FacialRecognition/FacialDetectionData/Disgust/')
os.mkdir('C:/Users/sbola/PycharmProjects/FacialRecognition/FacialDetectionData/Fear/')
os.mkdir('C:/Users/sbola/PycharmProjects/FacialRecognition/FacialDetectionData/Happy')
os.mkdir('C:/Users/sbola/PycharmProjects/FacialRecognition/FacialDetectionData/Sad/')
os.mkdir('C:/Users/sbola/PycharmProjects/FacialRecognition/FacialDetectionData/Surprise/')
os.mkdir('C:/Users/sbola/PycharmProjects/FacialRecognition/FacialDetectionData/Neutral/')

d = 0
for image_pixels in df0.iloc[1:, 1]:
    image_string = image_pixels.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48, 48)
    img = Image.fromarray(image_data)
    img.save('C:/Users/sbola/PycharmProjects/FacialDect/Angry/img_%d.jpg'%d, 'JPEG')
    d += 1

d = 0
for image_pixels in df1.iloc[1:, 1]:
    image_string = image_pixels.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48, 48)
    img = Image.fromarray(image_data)
    img.save('C:/Users/sbola/PycharmProjects/FacialDect/Disgust/img_%d.jpg'%d, 'JPEG')
    d += 1

d = 0
for image_pixels in df2.iloc[1:, 1]:
    image_string = image_pixels.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48, 48)
    img = Image.fromarray(image_data)
    img.save('C:/Users/sbola/PycharmProjects/FacialDect/Fear/img_%d.jpg'%d, 'JPEG')
    d += 1

d = 0
for image_pixels in df3.iloc[1:, 1]:
    image_string = image_pixels.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48, 48)
    img = Image.fromarray(image_data)
    img.save('C:/Users/sbola/PycharmProjects/FacialDect/Happy/img_%d.jpg'%d, 'JPEG')
    d += 1

d = 0
for image_pixels in df4.iloc[1:, 1]:
    image_string = image_pixels.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48, 48)
    img = Image.fromarray(image_data)
    img.save('C:/Users/sbola/PycharmProjects/FacialDect/Sad/img_%d.jpg'%d, 'JPEG')
    d += 1

d = 0
for image_pixels in df5.iloc[1:, 1]:
    image_string = image_pixels.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48, 48)
    img = Image.fromarray(image_data)
    img.save('C:/Users/sbola/PycharmProjects/FacialDect/Surprise/img_%d.jpg'%d, 'JPEG')
    d += 1

d = 0
for image_pixels in df6.iloc[1:, 1]:
    image_string = image_pixels.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48, 48)
    img = Image.fromarray(image_data)
    img.save('C:/Users/sbola/PycharmProjects/FacialDect/Neutral/img_%d.jpg'%d, 'JPEG')
    d += 1
