'''The following code trains the model'''

#importing relevant packages
from __future__ import absolute_import, division, print_function
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras import optimizers
import seaborn as sns
%matplotlib inline

## Import Keras objects for Deep Learning
from keras.models  import Sequential, K
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop

#importing the training file
df = pd.read_csv("/path_to_csv.csv/")

x = df.drop('Output', axis=1)
y = df['Output']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1111)

model_1 = Sequential([
	Dense(4, input_shape=(4,), activation='tanh'),
	Dense(8, activation='tanh'),
	Dense(8, activation='tanh'),
	Dense(1, activation='sigmoid')])
	
print("The model summary for the developed model is :: ", model_1.summary)

model_1.save("path_to_save_model/model.h5")

adam = optimizers.Adam(lr=0.001)
model_1.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
run_1 = model_1.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=500)

#plotting the training loss and accuracy
plt.figure(figsize=(18,10))
plt.plot(run_1.history['loss'], 'r', label="Training Loss")
plt.plot(run_1.history['acc'], 'b', label="Train Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Percentage %")
plt.legend()
plt.savefig("/path_to_save_image.png", dpi=300)

#plotting the validation loss and accuracy
plt.figure(figsize=(18,10))
plt.plot(run_1.history['val_loss'], 'r', label="Validation Loss")
plt.plot(run_1.history['val_acc'], 'b', label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Percentage %")
plt.legend()
plt.savefig("/path_to_save_image.png", dpi=300)

#evaluating the model on train and test set
print("Train split:")
loss, accuracy = model_1.evaluate(x_train, y_train, verbose=1)
print("Accuracy : {:5.2f}".format(accuracy))

print("Test split:")
loss, accuracy = model_1.evaluate(x_test, y_test, verbose=1)
print("Accuracy : {:5.2f}".format(accuracy))

predict_class = model_1.predict_classes(x_test)
print("Predicted Classes are :: ", predict_class)

predict_proba = model_1.predict(x_test)
print("Predicted Probability is :: ", predict_proba)
