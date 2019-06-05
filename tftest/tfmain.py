import tensorflow as tf
import glob
import argparse
import datetime
import tensorflow.contrib.eager as tfe
import time
import os

tf.enable_eager_execution()

parser = argparse.ArgumentParser(description='SpriteGAN')
parser.add_argument('--path_to_dataset', type=str, default='./', help='Path to dataset(default ./)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size(default 32)')
parser.add_argument('--gradient_updates', type=int, default=3500, help='Number of gradient updates(default 3500)')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning Rate(default 0.001)')
parser.add_argument('--output_dir', type=str, default='./', help='Output directory (default ./)')
args = parser.parse_args()

class Discriminator(tf.keras.Model):
  def __init__(self):
    super(Discriminator, self).__init__()  
    self.conv0 = tf.keras.layers.Conv2D(filters=32, kernel_size = 4, strides = 2, input_shape=(32, 32, 3), padding="VALID")
    self.batch_norm_0 = tf.keras.layers.BatchNormalization()

    self.conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size = 4, strides = 2, padding="VALID")
    self.batch_norm_1 = tf.keras.layers.BatchNormalization()

    self.conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size = 4, strides = 2)
    self.batch_norm_2 = tf.keras.layers.BatchNormalization()

    self.fully_connected = tf.keras.layers.Dense(1)
    self.flatten = tf.keras.layers.Flatten()

  def call(self, x):

    x = self.conv0(x)
    x = self.batch_norm_0(x)
    x = tf.nn.relu(x)

    x = self.conv1(x)
    x = self.batch_norm_1(x)
    x = tf.nn.relu(x)

    x = self.conv2(x)
    x = self.batch_norm_2(x)

    x = self.flatten(x) # 8*2*2
    x = self.fully_connected(x)
    x = tf.nn.sigmoid(x)
    return x   

class Generator(tf.keras.Model):
  def __init__(self):
    super(Generator, self).__init__()  
    self.fully_connected = tf.keras.layers.Dense(4*4*16, use_bias = False, input_shape=(100,))
    self.batch_norm_fc = tf.keras.layers.BatchNormalization()

    self.conv0 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size = 2, strides = 2, padding="VALID")
    self.batch_norm_0 = tf.keras.layers.BatchNormalization()

    self.conv1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size = 2, strides = 2, padding="VALID")
    self.batch_norm_1 = tf.keras.layers.BatchNormalization()

    self.conv2 = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size = 2, strides = 2, padding="VALID")
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

    x = tf.nn.tanh(x)
    return x

def convert_img_for_display(gen_x):
    OldMin = -1
    OldMax = 1
    NewMin = 0
    NewMax = 1
    OldRange = (OldMax - OldMin)
    NewRange = (NewMax - NewMin)
    img = ((gen_x - OldMin)*NewRange/OldRange+NewMin)
    return img

def convert_img_for_training(gen_x):
    OldMin = 0
    OldMax = 255
    NewMin = -1
    NewMax = 1
    OldRange = (OldMax - OldMin)
    NewRange = (NewMax - NewMin)
    img = ((gen_x - OldMin)*NewRange/OldRange+NewMin)
    return img

batch_size = args.batch_size
gradient_updates = 100000
draw_step = 500

zdim = 100
disc = Discriminator()
gen = Generator()
optimizer_disc = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
optimizer_gen = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

def _parse_function(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_gif(image_string)
  image_resized = tf.image.resize_images(image_decoded, [32, 32])
  image_new_range = convert_img_for_training(image_resized)
  return image_new_range

filenames = []
# Get file names
for filename in glob.glob(args.path_to_dataset+'/*.gif'):
    filenames.append(filename)

n = len(filenames)
# A vector of filenames.
#filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])
filenames = tf.constant(filenames)

dataset = tf.data.Dataset.from_tensor_slices(filenames).repeat()
dataset = dataset.map(_parse_function)
batched_dataset = dataset.batch(batch_size) 
# Redundant when tf.enable_eager_execution() enabled, or maybe only in 1.8
iterator = tfe.Iterator(batched_dataset)

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')
output_dir = args.output_dir + st + "sprite/"
os.mkdir(output_dir)
summary_writer = tf.contrib.summary.create_file_writer(output_dir)

random_vector_for_generation = tf.random_normal([8,zdim])

with tf.Session() as sess, summary_writer.as_default():

    track_d_loss = []
    track_g_loss = []
    for step in range(gradient_updates):
        start = datetime.datetime.now()
        try:
            batch_xs = iterator.get_next()
        except tf.errors.OutOfRangeError:
            iterator = tfe.Iterator(batched_dataset)
            continue
        # Note: decode_gif returns a 4-D array, remove this extra dim which indicates nr of frames in gif
        batch_xs = tf.squeeze(batch_xs)
        # Record operations for automatic differentiation.
        with tfe.GradientTape() as disc_tape, tfe.GradientTape() as gen_tape:
          # Train Discriminator
          y_real = disc(batch_xs)
          # Real data : label 1
          y_real_label = tf.ones(tf.shape(y_real))
          loss_real = tf.keras.backend.binary_crossentropy(y_real_label, y_real)

          z = tf.random_normal(shape=(batch_size, 100), dtype='float32')
          gen_x = tf.stop_gradient(gen(z))
          y_generated = disc(gen_x)
          y_gen_label = tf.zeros(tf.shape(y_generated))
          loss_generated = tf.keras.backend.binary_crossentropy(y_generated, y_gen_label)
          
          loss_disc = loss_real + loss_generated
          loss_disc = tf.reduce_sum(loss_disc)
          
          # Train Generator
          z_gen = tf.random_normal(shape=(batch_size, 100), dtype='float32')
          batch_xs_gen = gen(z_gen)
          y_gen = disc(batch_xs_gen)
          # generator is trying to generate fake images that resemble the real images -> label = 1
          y_gen_label = tf.ones(tf.shape(y_gen)) 
          loss_gen = tf.keras.backend.binary_crossentropy(y_gen_label, y_gen)
          loss_gen = tf.reduce_sum(loss_gen)

        # Calculate gradients
        grads_disc = disc_tape.gradient(loss_disc, disc.trainable_variables)
        optimizer_disc.apply_gradients(zip(grads_disc, disc.trainable_variables))
        print("Step: {},         Disc Loss: {}".format(step, loss_disc.numpy()))

        grads_gen = gen_tape.gradient(loss_gen, gen.trainable_variables)
        optimizer_gen.apply_gradients(zip(grads_gen, gen.trainable_variables))
        print("Step: {},         Gen Loss: {}".format(step, loss_gen.numpy()))
        tf.contrib.summary.scalar("loss/gen",loss_gen.numpy(), step)
        # Use same latent vector to see progress
        gen_x_log = tf.stop_gradient(gen(random_vector_for_generation))


""" with tf.GradientTape(watch_accessed_variables=False) as tape:
  tape.watch(a.variables)  # Since `a.build` has not been called at this point
                           # `a.variables` will return an empty list and the
                           # tape will not be watching anything.
  result = b(a(inputs))
  tape.gradient(result, a.variables)  # The result of this computation will be
                                      # a list of `None`s since a's variables
                                      # are not being watched.
 """