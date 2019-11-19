import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
from typing import Dict, Tuple, Sequence, List

class vanila_neural_net:
  """
  A simple neural network
  
  Attributes
  ----------
  input_sz: int
    the number of neurons in the input layer

  output_sz: int 
    the number of neurons in the output layer

  input_dim: int 
    the size of the input vector
    
  Methods
    model: layers of the simple neural network
  
  """
  def __init__(self, input_sz, output_sz, input_dim, alpha, num_hidden_layer=1, dropout=0.0) -> None:
    self.input_sz = input_sz
    self.output_sz = output_sz
    self.input_dim = input_dim
    self.num_hidden_layer = num_hidden_layer
    self.alpha = alpha
    self.dropout = dropout

    # self.q_network = self.init_model(tf.nn.relu)
    self.optimiser = tf.keras.optimizers.Adam(learning_rate=self.alpha)

  def init_model(self, activation_func="a", init_way:str="glorot_normal") -> None:
    model = models.Sequential()
    model.add(layers.Dense( self.input_sz, activation='relu', input_dim=self.input_sz))
    model.add(layers.Dense( self.output_sz, activation='relu'))
    model.compile(optimizer=optimizers.Adam(lr=self.alpha),
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    """ 
    model = tf.keras.Sequential([
      tf.keras.layers.Dense(
        self.input_sz, 
        activation=activation_func,
        kernel_initializer=init_way,
        input_dim=self.input_sz
      ),
      tf.keras.layers.Dropout(self.dropout), 
      tf.keras.layers.Dense(
        self.output_sz, 
        kernel_initializer=init_way,
        activation=activation_func
       )
    ])
    """
    return model
