class vanila_neural_net:
  def __init__(self, input_sz, output_sz, input_dim, num_hidden_layer=1):
    
    self.input_sz = input_sz
    self.output_sz = output_sz
    self.input_dim = input_dim
    self.num_hidden_layer = num_hidden_layer

  def model(self):
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(input_dim)),
      tf.keras.layers.Dense(input_sz, activation='relu'),
      # tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(output_sz, activation='softmax')
    ])
    return model


class lstm_neural_net:
  def __init__(self, input_sz, output_sz, input_dim, num_hidden_layer=1, dpr=0.0):
    self.num_hidden_layer = num_hidden_layer
    self.lstm1 = tf.keras.layers.LSTM(
      input_sz, 
      kernel_initializer="glorot_uniform",
      input_shape=input_dim
    )
    self.ln1 = tf.keras.layers.LayerNormalization()
    self.dpL = tf.keras.layers.Dropout(dpr)
​
​
    self.out = tf.keras.layers.Dense(
      output_sz, 
      kernel_initializer="glorot_normal"
    )
    self.ln2 = tf.keras.layers.LayerNormalization()

  def model(self):
    model = tf.keras.models.Sequential([
      tf.keras.layers.LSTM(input_sz, kernel_initializer="glorot_uniform",input_shape=input_dim)
      tf.keras.layers.Dropout(dpr)
      self.ln1,
      self.out,
      self.ln2
    ])

    return model