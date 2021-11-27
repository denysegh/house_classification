# https://colab.research.google.com/github/keras-team/autokeras/blob/master/docs/ipynb/structured_data_classification.ipynb#scrollTo=O4A-CqeBPYf2
!pip install autokeras

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

import autokeras as ak

# 1) load data 
train_file_path = "/Users/sabrinazhang/Documents/hhb_ML/data.csv"

# x_train as pandas.DataFrame, y_train as pandas.Series
x_train = pd.read_csv(train_file_path)
print(type(x_train))  # pandas.DataFrame
y_train = x_train.pop("status")
print(type(y_train))  # pandas.Series
# You can also use pandas.DataFrame for y_train.
y_train = pd.DataFrame(y_train)
print(type(y_train))  # pandas.DataFrame
# You can also use numpy.ndarray for x_train and y_train.
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
print(type(x_train))  # numpy.ndarray
print(type(y_train))  # numpy.ndarray

# 2) build and train models by using autokeras
# It tries 10 different models.
clf = ak.StructuredDataClassifier(overwrite=True, max_trials=10)
# Feed the structured data classifier with training data.
clf.fit(x_train, y_train, 
        validation_split=0.15, 
        epochs=10) # 10 epoches / model
        
# 3) export the best model   
model = clf.export_model()
model.summary()
print(x_train.dtype)
# numpy array in object (mixed type) is not supported.
# convert it to unicode.
model.predict(x_train.astype(np.unicode))

try:
    model.save("model_autokeras", save_format="tf")
except Exception:
    model.save("model_autokeras.h5")

# 4) use the saved the model
loaded_model = load_model("model_autokeras", custom_objects=ak.CUSTOM_OBJECTS)
#predicted_y = loaded_model.predict(tf.expand_dims(x_test, -1))
#print(predicted_y)
