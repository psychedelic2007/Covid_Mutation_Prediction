'''Predicting on test data'''

#importing relevant packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

new_data = pd.read_csv("/path_to_new_data.csv")

x_predict = new_data.drop('Output', axis=1)
y_predict = new_data['Output']

predict_prob = model_1.predict(x_predict)
print(predict_prob)

plt.figure(figsize=(18,10))
plt.plot(predict_prob[:,0])
plt.xlabel("Residue")
plt.ylabel("Probability")
plt.show()
