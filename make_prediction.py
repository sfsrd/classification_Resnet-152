from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image

class_names = ['cat', 'dog']

MODEL_FILE = 'train_resnet152/model_ResNet152_25.model'
IMAGE_PATH = 'train_resnet152/cat_test.jpg'
model = load_model(MODEL_FILE, compile = True)
image_size = 128

img = keras.preprocessing.image.load_img(
    IMAGE_PATH, target_size=(image_size, image_size)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

line = '{} with a {:.2f} percent confidence.'.format(class_names[np.argmax(score)], 100 * np.max(score))

image = image.imread(IMAGE_PATH)
plt.figure(figsize=(5, 5))

plt.imshow(image)
plt.title(line)
plt.axis("off")
plt.savefig("train_resnet152/result.png")
plt.show()