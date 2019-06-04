import tensorflow as tf
import glob
import argparse
import datetime
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

parser = argparse.ArgumentParser(description='SpriteGAN')
parser.add_argument('--path_to_dataset', type=str, default='./', help='Path to dataset(default ./)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size(default 32)')
parser.add_argument('--gradient_updates', type=int, default=3500, help='Number of gradient updates(default 3500)')
args = parser.parse_args()

class Discriminator(tf.keras.Model):
  def __init__(self):
    super(Discriminator, self).__init__()  
    # Changed data_format = 'channels_first' in keras.json file, different from default
    self.conv0 = tf.keras.layers.Conv2D(filters=32, kernel_size = 4, strides = 2, input_shape=(3, 32, 32), padding="VALID")
    self.batch_norm_0 = tf.keras.layers.BatchNormalization()

    self.conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size = 4, strides = 2, padding="VALID")
    self.batch_norm_1 = tf.keras.layers.BatchNormalization()

    self.conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size = 4, strides = 2)
    self.batch_norm_2 = tf.keras.layers.BatchNormalization()

    self.fully_connected = tf.keras.layers.Dense(1, input_shape = (8*4*4,))


  def call(self, x):
    x = self.conv0(x)
    x = self.batch_norm_0(x)
    x = tf.nn.relu(x)

    x = self.conv1(x)
    x = self.batch_norm_1(x)
    x = tf.nn.relu(x)

    x = self.conv2(x)
    x = self.batch_norm_2(x)

    x = tf.keras.layers.Reshape(x, target_shape = (8*4*4))
    x = self.fully_connected(x)
    x = tf.nn.sigmoid(x)
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
device = '/cpu:0'

zdim = 100
disc = Discriminator()

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

with tf.Session() as sess:
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
        s = disc(batch_xs)
        print(s)
        d = input("Pause eller?")


""" with tf.GradientTape(watch_accessed_variables=False) as tape:
  tape.watch(a.variables)  # Since `a.build` has not been called at this point
                           # `a.variables` will return an empty list and the
                           # tape will not be watching anything.
  result = b(a(inputs))
  tape.gradient(result, a.variables)  # The result of this computation will be
                                      # a list of `None`s since a's variables
                                      # are not being watched.
 """