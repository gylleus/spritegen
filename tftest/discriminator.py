import tensorflow as tf

class Discriminator(tf.keras.Model):
  def __init__(self):
    super(Discriminator, self).__init__()  
    self.conv0 = tf.keras.layers.Conv2D(filters=8, kernel_size = 4, strides = 2, input_shape=(32, 32, 3), padding="VALID")
    self.batch_norm_0 = tf.keras.layers.BatchNormalization()

    self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size = 4, strides = 2, padding="VALID")
    self.batch_norm_1 = tf.keras.layers.BatchNormalization()

    self.conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size = 4, strides = 2)
    self.batch_norm_2 = tf.keras.layers.BatchNormalization()

    self.fully_connected = tf.keras.layers.Dense(1)
    self.flatten = tf.keras.layers.Flatten()

  def call(self, x):

    x = self.conv0(x)
    x = self.batch_norm_0(x)
    x = tf.nn.leaky_relu(x)

    x = self.conv1(x)
    x = self.batch_norm_1(x)
    x = tf.nn.leaky_relu(x)

    x = self.conv2(x)
    x = self.batch_norm_2(x)
    x = tf.nn.leaky_relu(x)

    x = self.flatten(x) # 8*2*2
    x = self.fully_connected(x)
    x = tf.nn.sigmoid(x)
    return x 