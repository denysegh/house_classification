# ref: https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/intro_to_keras_for_engineers.ipynb#scrollTo=VN3kKKJhg4gW
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

''' # 0) Provide constants, prior info and constraints
 > the list of to-remove features
 > the list of the categorical features
 > labels' names 
'''
NUM_CLASSES = 10


''' # 1) Load raw data along with labels
 > CSV data needs to be parsed, with numerical features converted to floating point tensors and categorical features indexed and converted to integer tensors. Then each feature typically needs to be normalized to zero-mean and unit-variance.
 > tf.data.experimental.make_csv_dataset to load structured data from CSV files.
'''


''' # 2) Preprocess raw data: 
 > inputs: NumPy arrays w/ lables
       w/ prior constraints provided in #0)
 > ouputs: Processed NumPy arrays
 > what to do:
    - feature selection w/ prior constraints
    - categorical feature conversion
    - feature normalization (do we need it here?)
    - feature scaling (within [0,1])
'''

''' # 3) Build the models
'''
# 3.1) Logistic regression: 

# 3.2) MLP:
def build_mlp_model:
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.summary()

# 4) Compile and train the model
model = build_mlp_model()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    run_eagerly=True)
# this eager execution is for debug only and will slow down the training process 
# it needs to be turned off after debug is done

# 5) Train the model
# checkpoint cb get callled after each epoch to store the intermediate info
cp_cb = [
    keras.callbacks.ModelCheckpoint(
        filepath='path/to/my/model_{epoch}',
        save_freq='epoch')
]
# also TensorBoard can be used to monitor training process 

num_epoches = 20
batch_size = 64
print("Fit on NumPy data")
history = model.fit(x_train, y_train, 
                    batch_size=batch_size, 
                    epochs=num_epoches,
                    callbacks=cp_cb)


# 6) Evaluate the model
loss, acc = model.evaluate(val_dataset)  # returns loss and metrics
print("loss: %.2f" % loss)
print("acc: %.2f" % acc)

# 7) Test
predictions = model.predict(val_dataset)
print(predictions.shape)

