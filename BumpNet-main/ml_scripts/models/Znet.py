import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Conv1D, Reshape
from tensorflow.keras.optimizers import Adam

class Znet(tf.keras.Model):

  def __init__(self,N_bins):
    super().__init__()

    self.model = Sequential()

    self.model.add(Dense(N_bins, activation='relu'))
    self.model.add(Reshape((N_bins,1)))

    kernels = [25,20,15,10,5,1]
    filters = [64,32,32,16,8,1]

    for i, (kern, filt) in enumerate(zip(kernels,filters)):
        act = 'relu' if i+1<len(kernels) else None
        self.model.add(Conv1D(kernel_size=kern, filters=filt, padding='same', activation=act))

  def call(self, inputs):

    outputs = self.model(inputs)
    return outputs
