import tensorflow as tf

class Generator(tf.keras.Model):
  def __init__(self):
    super(Generator, self).__init__()  
    self.fully_connected = tf.keras.layers.Dense(4*4*16, use_bias = False, input_shape=(100,))
    #self.batch_norm_fc = tf.keras.layers.BatchNormalization()
    # BN eliminates the bias, and BN (with affine=True) adds a bias so it’s equivalent.

    self.conv0 = tf.keras.layers.Conv2D(filters=16, kernel_size = 2, strides = 1, padding="SAME")
    #self.batch_norm_0 = tf.keras.layers.BatchNormalization()

    self.conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size = 3, strides = 1, padding="SAME")
    #self.batch_norm_1 = tf.keras.layers.BatchNormalization()

    # Random size of kernel and stride
    self.conv2 = tf.keras.layers.Conv2D(filters=3, kernel_size = 4, strides = 1, padding="SAME")
    #self.batch_norm_2 = tf.keras.layers.BatchNormalization()

  def call(self, z):

    x = self.fully_connected(z)
    #x = self.batch_norm_fc(x)
    x = tf.nn.relu(x)
    x = tf.reshape(x, [-1, 4, 4, 16])
    x = tf.image.resize_bicubic(x, [8, 8])
    x = self.conv0(x)
    #x = self.batch_norm_0(x)
    x = tf.nn.relu(x)
    x = tf.image.resize_bicubic(x, size=[16, 16])
    x = self.conv1(x)
    #x = self.batch_norm_1(x)
    x = tf.nn.relu(x)
    x = tf.image.resize_bicubic(x, size=[32, 32])
    x = self.conv2(x)
    #x = self.batch_norm_2(x)
    x = tf.nn.tanh(x, name="output_node")
    return x