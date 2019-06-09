import tensorflow as tf

class Generator(tf.keras.Model):
  def __init__(self):
    super(Generator, self).__init__()  
    self.fully_connected = tf.keras.layers.Dense(4*4*16, use_bias = False, input_shape=(100,))
    self.batch_norm_fc = tf.keras.layers.BatchNormalization()
    # BN eliminates the bias, and BN (with affine=True) adds a bias so it’s equivalent.

    self.conv0 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size = 2, strides = 2, padding="VALID", use_bias = False)
    self.batch_norm_0 = tf.keras.layers.BatchNormalization()

    self.conv1 = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size = 2, strides = 2, padding="VALID", use_bias = False)
    self.batch_norm_1 = tf.keras.layers.BatchNormalization()

    self.conv2 = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size = 2, strides = 2, padding="VALID", use_bias = False)
    self.batch_norm_2 = tf.keras.layers.BatchNormalization()

  def call(self, z):

    x = self.fully_connected(z)
    x = self.batch_norm_fc(x)
    x = tf.nn.relu(x)
    x = tf.reshape(x, [-1, 4, 4, 16])

    x = self.conv0(x)
    x = self.batch_norm_0(x)
    x = tf.nn.relu(x)

    x = self.conv1(x)
    x = self.batch_norm_1(x)
    x = tf.nn.relu(x)

    x = self.conv2(x)
    x = self.batch_norm_2(x)

    x = tf.nn.tanh(x, name="output_node")
    return x