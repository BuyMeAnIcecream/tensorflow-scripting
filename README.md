# tensorflow-scripting
Py scripts using tensorflow for various ML tasks


So I've been running different machine learning tests in this one. The set up is real finicky. Here is what I'm on currently. 

py3.8

venv
https://docs.python.org/3/library/venv.html

tensorflow
https://www.tensorflow.org/install/pip#virtual-environment-install

jupyter lab
https://jupyter.org/install



GPU
cuDNN 7.6.5
CUDA Toolkit 10.1 
!!!I couldn't hook up all the necessary CUDA dlls with PATH set up so I had to copy them to Windows\System32 


DATA
lungs:
https://www.kaggle.com/kostasdiamantaras/chest-xrays-bacterial-viral-pneumonia-normal?select=labels_train.csv


cats and dogs data set
https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data?select=train.zip

theoretical_drug_test - bunch of syntax. putting together first sequential network. loading, saving the model.

dogs_and_cats - images. teaching network to tell 2 classes and using pretrained VGG16 to classify only those

pneumonia_detector - now we have bunch of lung scans. Healthy lungs and with viral and bacterial pneumonia. 