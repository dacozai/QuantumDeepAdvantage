import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers

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
  def __init__(self, input_sz, output_sz, input_dim, alpha, num_hidden_layer=1):
  
    self.input_sz = input_sz
    self.output_sz = output_sz
    self.input_dim = input_dim
    self.num_hidden_layer = num_hidden_layer
    self.alpha = alpha

  def init_model(self):
    model = models.Sequential()
    model.add(layers.Dense( self.input_sz, activation='relu', input_dim=self.input_sz))
    model.add(layers.Dense( self.input_sz, activation='relu'))
    model.add(layers.Dense( self.output_sz, activation='relu'))
    model.compile(optimizer=optimizers.Adam(lr=self.alpha),
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    return model
