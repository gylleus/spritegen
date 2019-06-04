import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import datetime
from cdgan import CDCGAN
import glob
import argparse

parser = argparse.ArgumentParser(description='SpriteGAN')
parser.add_argument('--path_to_dataset', type=str, default='.', help='Path to dataset')
args = parser.parse_args()

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



batch_size = 32#8
gradient_updates = 100000
draw_step = 500
device = '/cpu:0'

zdim = 100
ydim = 1

#mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_gif(image_string)
  image_resized = tf.image.resize_images(image_decoded, [32, 32])
  return image_resized, label

filenames = []
# Get file names
for filename in glob.glob(args.path_to_dataset+'/*.gif'):
    filenames.append(filename)

n = len(filenames)
# A vector of filenames.
#filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])
filenames = tf.constant(filenames)
print(filenames)
# `labels[i]` is the label for the image in `filenames[i].
labels = np.ones((n, 1))
labels = tf.constant(labels) # label 0 for character images

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels)).repeat()
dataset = dataset.map(_parse_function)
batched_dataset = dataset.batch(batch_size) 
dataset = batched_dataset
iterator = batched_dataset.make_one_shot_iterator()

with tf.device(device):
    model = CDCGAN(zdim, ydim, [32, 32, 3])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    track_d_loss = []
    track_g_loss = []
    for step in range(gradient_updates):
        start = datetime.datetime.now()
        try:
            batch_xs, batch_ys = iterator.get_next()
        except tf.errors.OutOfRangeError:
            iterator = batched_dataset.make_one_shot_iterator()
            continue
        #print(zdim.size())
        gloss, dloss = model.train_step(sess, convert_img_for_training(batch_xs.eval()[:,0,:,:,:]),
                batch_ys.eval(), np.random.uniform(0, 1, (batch_size, zdim)), np.ones((batch_size, 1)))
        track_d_loss.append(dloss)
        track_g_loss.append(gloss) # self, sess, xs, d_ys, zs, g_ys, is_training=True
        if step < 1000:
            sched = 300
        else:
            sched = 300
        if step % sched == 0:                      # zs, ys, is_training
            imgs = model.sample_generator(sess, zs=np.repeat(np.random.uniform(-1, 1, (10, zdim)), 10, axis=0),
                                          ys=np.repeat(np.ones((10, 1)), 10, axis=0))#np.tile(np.eye(ydim), [1, 1]))
            fig = plt.figure()
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0.1)
            for i in range(10*10):
                fig.add_subplot(10, 10, i + 1)
                #testar = tf.check_numerics(imgs[i, :, :, :], "NAN FOUND!!!!", name=None)
                #print('Vad tsta; ', testar)
                plt.imshow(convert_img_for_display(imgs[i, :, :, :]))
                plt.axis('off')
                # kan det vara vara att NaN+värden genereras?
            plt.savefig('./GeneratedImages/iter_{}.png'.format(step))
            plt.close()
            end = datetime.datetime.now()
            print('step: {}/{},\t G loss: {:.4f}, D loss: {:.4f}\t| time: {}'.
                  format(step, gradient_updates, gloss, dloss, end - start))
            # Then plot the loss in an appealing format
            if step > 1: # Ensures there is data to plot
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
                plt.savefig('./GeneratedImages/images'+'loss_tracking.png')
                plt.close()

