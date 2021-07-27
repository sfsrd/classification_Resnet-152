import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

# if you have GPU - uncomment it
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print("physical_devices-------------", len(physical_devices), ': ', physical_devices)
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# number of classes you train NN for
NUM_CLASSES = 2
# rgb - 3; black&white - 1
CHANNELS = 3
# the image's resolution for training
IMAGE_RESIZE = 128
# paraneters for training
RESNET152_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'
LOSS_METRICS = ['accuracy']
NUM_EPOCHS = 3
EARLY_STOP_PATIENCE = 3
STEPS_PER_EPOCH_TRAINING = 10
STEPS_PER_EPOCH_VALIDATION = 10
BATCH_SIZE_TRAINING = 50
BATCH_SIZE_VALIDATION = 50
BATCH_SIZE_TESTING = 1

def plotting(fit_history):
    plt.plot(fit_history.history['accuracy'], label = 'train_acc')
    plt.plot(fit_history.history['val_accuracy'], label = 'val_accuracy')
    plt.plot(fit_history.history['loss'], label = 'train_loss')
    plt.plot(fit_history.history['val_loss'], label = 'val_loss')
    plt.title('model acc / loss')
    plt.ylabel('loss / acc')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.savefig("D:/Personal/selfeducation/classification/train_resnet152/ResNet152_training.png")
    plt.show()

def load_datasets():
    image_size = IMAGE_RESIZE
    data_generator = ImageDataGenerator()

    train_generator = data_generator.flow_from_directory(
            'D:/Personal/selfeducation/classification/train_resnet152/dataset/train/',
            target_size=(image_size, image_size),
            batch_size=BATCH_SIZE_TRAINING,
            class_mode='categorical')

    validation_generator = data_generator.flow_from_directory(
            'D:/Personal/selfeducation/classification/train_resnet152/dataset/test/',
            target_size=(image_size, image_size),
            batch_size=BATCH_SIZE_VALIDATION,
            class_mode='categorical')
    
    return train_generator, validation_generator

def define_architecture():
    model = Sequential()
    model.add(ResNet152(include_top = False, pooling = RESNET152_POOLING_AVERAGE, weights = "imagenet"))
    # add the dense with NUM_CLASSES as number of uotputs
    model.add(Dense(NUM_CLASSES, activation = DENSE_LAYER_ACTIVATION))
    model.layers[0].trainable = False
    model.summary()
    sgd = optimizers.SGD(learning_rate = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
    model.compile(optimizer = sgd, loss = OBJECTIVE_FUNCTION, metrics = LOSS_METRICS)

    return model

def training(model):
    fit_history = model.fit(
            train_generator,
            steps_per_epoch=STEPS_PER_EPOCH_TRAINING,
            epochs = NUM_EPOCHS,
            validation_data=validation_generator,
            validation_steps=STEPS_PER_EPOCH_VALIDATION
    )
    model.save('D:/Personal/selfeducation/classification/train_resnet152/model_ResNet152_25.model')

    return fit_history

# step 1 - flow datasets
train_generator, validation_generator = load_datasets()

# step 2 - define architecture of NN including parameters
model = define_architecture()

#step 3 - training
fit_history = training(model)

# step 4 - see the graphic
plotting(fit_history)