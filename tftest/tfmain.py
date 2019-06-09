from discriminator import Discriminator
from generator import Generator
import model_saver as ms
#from tensorboardX import SummaryWriter # ...
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import glob
import argparse
import datetime
import tensorflow.contrib.eager as tfe
import time
import os

parser = argparse.ArgumentParser(description='SpriteGAN')
parser.add_argument('--path_to_dataset', type=str, default='./', help='Path to dataset(default ./)')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size(default 64)')
parser.add_argument('--gradient_updates', type=int, default=3500, help='Number of gradient updates(default 3500)')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning Rate(default 0.0002)')
parser.add_argument('--output_dir', type=str, default='./', help='Output directory (default ./)')
parser.add_argument('--save_img', type=int, default=500, help='Save img interval (default 500)')
parser.add_argument('--load_models', type=int, default=0, help='Load a pre-trained model. True: 1, False: 0 (default 0)')
parser.add_argument('--saved_models_dir', type=str, default='./', help='Path to saved models (default ./)')
parser.add_argument('--checkpoint', type=int, default=20000, help='Save model after set number of steps (default 20000)')
parser.add_argument('--export_model', action='store_true')
args = parser.parse_args()

if args.export_model:
  ms.save_generator(args.saved_models_dir + "generator")
  exit()

tf.enable_eager_execution()

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
lamb = 10

zdim = 100
# Define learning rate as variable in order to decay it later
starter_learning_rate = args.learning_rate
learning_rate = tfe.Variable(starter_learning_rate)
disc = Discriminator()
gen = Generator()
optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)

if args.load_models:
  # retarded men fungerar - enligt https://github.com/keras-team/keras/issues/10417 , Ketan14as kommentar
  zass = tf.random_normal(shape=(batch_size, 100), dtype='float32')
  dafuq_data = tf.stop_gradient(gen(zass))
  sauce = disc(dafuq_data)
  disc.load_weights(args.saved_models_dir + "discriminator")
  gen.load_weights(args.saved_models_dir + "generator")

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
output_dir_fixed = args.output_dir + st + "sprite/fixed/"
output_dir_rand = args.output_dir + st + "sprite/rand/"
output_dir_model = args.output_dir + st + "sprite/models/"
os.mkdir(output_dir)
os.mkdir(output_dir_fixed)
os.mkdir(output_dir_rand)
os.mkdir(output_dir_model)

random_vector_for_generation = tf.random_normal([16,zdim])

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
    z = tf.random_normal(shape=(batch_size, 100), dtype='float32')
    gen_x = tf.stop_gradient(gen(z))
    # Gradient Pentalty
    gp = 0
    """         
    epsilon = tf.random_uniform([], 0, 1)
    xhat = epsilon*batch_xs + (1-epsilon)*gen_x
    with tfe.GradientTape() as gtape:
        gtape.watch(xhat)
        dhat = disc(xhat)
    dhat2 = gtape.gradient(dhat, xhat)
    slopes = tf.sqrt(tf.reduce_sum(tf.square(dhat2), reduction_indices=[1]))
    gp = tf.reduce_mean((slopes-1.0)**2) 
    """

    # Record operations for automatic differentiation.
    with tfe.GradientTape() as disc_tape, tfe.GradientTape() as gen_tape:
      # Train Discriminator
      y_real = disc(batch_xs)
      # Real data : label 1
      y_real_label = tf.ones(tf.shape(y_real))
      loss_real = tf.keras.backend.binary_crossentropy(y_real_label, y_real)

      y_generated = disc(gen_x)
      y_gen_label = tf.zeros(tf.shape(y_generated))
      loss_generated = tf.keras.backend.binary_crossentropy(y_generated, y_gen_label)

      loss_disc = loss_real + loss_generated + lamb*gp
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

    grads_gen = gen_tape.gradient(loss_gen, gen.trainable_variables)
    optimizer_gen.apply_gradients(zip(grads_gen, gen.trainable_variables))

    track_d_loss.append(loss_disc.numpy())
    track_g_loss.append(loss_gen.numpy())
    if step > 10000:
      # Decay learning rate
      learning_rate.assign(tf.train.exponential_decay(starter_learning_rate, step-10000, decay_steps=30000, decay_rate=0.8))
    #writer.add_scalar("loss/crit", loss_disc.numpy(), step)
    #writer.add_scalar("loss/gen", loss_gen.numpy(), step)
    if step % args.checkpoint == 0:
      disc.save_weights(output_dir_model+'discriminator')
      gen.save_weights(output_dir_model+'generator')
    
    if step % args.save_img == 0:

      print("Learning rate: ", learning_rate)
      # Use same latent vector to see progress
      gen_x_log = tf.stop_gradient(gen(random_vector_for_generation))
      gen_img = gen_x_log.numpy()
      print("Step: ", step)
      z_gen = tf.random_normal(shape=(batch_size, 100), dtype='float32')
      gen_img_rand = gen(z_gen)
      #gen_img = np.transpose(gen_img, (0, 3, 1, 2))
      #writer.add_image(str(step), gen_img[0, :, :, :], step)

      # temp, tensorboard just doesn't wont to cooperate
      def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

      xs = [_ for _ in range(len(track_d_loss))]
      w = 10
      f, axs = plt.subplots(2, figsize=(8,4), sharex=True)
      plt.xkcd()
      axs[0].plot(track_d_loss, alpha=0.3, linewidth=5)
      axs[0].plot(xs[w:-w], smooth(track_d_loss, w)[w:-w], c='C0')
      axs[0].set_title('Discriminator', fontsize=10)
      axs[0].set_yscale('log')
      axs[0].set_ylabel('loss', fontsize=10)
      plt.xkcd()
      axs[1].plot(track_g_loss, alpha=0.3, linewidth=5, c='C4')
      axs[1].plot(xs[w:-w], smooth(track_g_loss, w)[w:-w], c='C4')
      axs[1].set_title('Generator', fontsize=10)
      axs[1].set_xlabel('step')
      axs[1].set_ylabel('loss', fontsize=10)
      plt.savefig(output_dir+'loss_tracking.png')
      plt.close()

      fig = plt.figure()
      fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0.1)
      for i in range(4*4):
          fig.add_subplot(4, 4, i + 1)
          plt.imshow(convert_img_for_display(gen_img[i, :, :, :]))
          plt.axis('off')
      plt.savefig(output_dir_fixed + 'iter_{}.png'.format(step))
      plt.close()

      fig = plt.figure()
      fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0.1)
      for i in range(4*4):
          fig.add_subplot(4, 4, i + 1)
          plt.imshow(convert_img_for_display(gen_img_rand[i, :, :, :]))
          plt.axis('off')
      plt.savefig(output_dir_rand + 'iter_{}.png'.format(step))
      plt.close()


#writer.close()

