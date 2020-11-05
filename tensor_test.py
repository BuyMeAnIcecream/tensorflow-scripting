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

predictions = model.predict(x = scaled_train_samples, batch_size = 10, verbose = 0)
#for i in predictions:
#    print(i)
    
rounded_prediction = np.argmax(predictions, axis = -1)

#for i in rounded_prediction:
#    print(i)


#PREPROCESS TEST DATA
test_labels = []
test_samples = []

for i in range(10):
    #the ~5% of younger individuals who did exp side eff
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(1)

    #the ~5% of older individuals who did not exp side eff
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(0)

for i in range(200):
    #the ~95% of younger individuals who did not exp side eff
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)

    #the ~95% of older individuals who did exp side eff
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)

test_samples = np.array(test_samples)
test_labels = np.array(test_labels)
test_labels, test_samples = shuffle(test_labels, test_samples);

scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1,1))

predictions = model.predict(x = scaled_test_samples, batch_size = 10, verbose = 0)
#for i in predictions:
#    print(i)
    
rounded_prediction = np.argmax(predictions, axis = -1)

#for i in rounded_prediction:
#    print(i)

#input("press close to exit") 

#CONFUSION MATRIX
%matplotlib inline
from sklearn.metrics import confusion_matrix
import itertools
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true = test_labels, y_pred = rounded_prediction)

def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = 'Cunfusion matrix',
                          cmap = plt.cm.Blues):
    """
    Prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize  = True'.
    """
    
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without norm")
        
    print(cm)
    
    thresh = cm.max() / 2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i,j],
            horizontalalignment = "center",
            color = "white" if cm[i,j] > thresh else "black")
            
            
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')
    

    
cm_plot_labels = ['no_side_effects', 'had_side_effects']
plot_confusion_matrix(cm = cm, classes = cm_plot_labels, title = 'Confusion Matrix')
  #  input("press close to exit") 