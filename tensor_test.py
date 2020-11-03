#PREPARE DATA
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

train_labels = []
train_samples = []

for i in range(50):
    #the ~5% of younger individuals who did exp side eff
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)

    #the ~5% of older individuals who did not exp side eff
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    #the ~95% of younger individuals who did not exp side eff
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    #the ~95% of older individuals who did exp side eff
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)

#for i in train_samples:
#    print(i)
train_samples = np.array(train_samples)
train_labels = np.array(train_labels)
train_labels, train_samples = shuffle(train_labels, train_samples);

scaler = MinMaxScaler(feature_range = (0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))

#for i in scaled_train_samples:
#    print(i)

input("press close to exit") 

#CONSTRUCT SEQUENTIAL NN


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

#TODO figure gpu bs
#physical_dev = tf.config.experimental.list_physical_devices('GPU')
#print("Num GPUs avail: ", len(physical_dev))
#tf.config.experimental.set_memory_growth(physical_dev[0], True)

model = Sequential([
    Dense(units = 16, input_shape = (1,), activation = 'relu'),
    Dense(units = 32, activation = 'relu'),
    Dense(units = 2, activation = 'softmax')
])

model.compile(optimizer = Adam(learning_rate = 0.0001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(x = scaled_train_samples, y = train_labels, validation_split = 0.1, batch_size = 10, epochs = 30, shuffle = True, verbose = 2)

 

model.summary()

