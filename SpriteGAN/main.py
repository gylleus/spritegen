import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import datetime
import argparse 
from cdcgan import CDCGAN

#mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

batch_size = 128
epochs = 100000
draw_step = 500
device = '/cpu:0'

zdim = 100
ydim = 10
print(tor.ss)

with tf.device(device):
    model = CDCGAN(zdim, ydim, [28, 28, 1])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    track_d_loss = []
    track_g_loss = []
    for epoch in range(epochs):
        start = datetime.datetime.now()
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        gloss, dloss = model.train_step(sess, np.reshape(2.*batch_xs - 1., [-1, 28, 28, 1]),
                        batch_ys, np.random.uniform(-1, 1, (batch_size, zdim)), batch_ys)
        track_d_loss.append(dloss)
        track_g_loss.append(gloss)
        if epoch < 1000:
            sched = 50
        else:
            sched = 250
        if epoch % sched == 0:
            imgs = model.sample_generator(sess, zs=np.repeat(np.random.uniform(-1, 1, (10, zdim)), 10, axis=0),
                                          ys=np.tile(np.eye(ydim), [10, 1]))
            fig = plt.figure()
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0.1)
            for i in range(10*10):
                fig.add_subplot(10, 10, i + 1)
                plt.imshow(imgs[i, :, :, 0], cmap='gray')
                plt.axis('off')
            plt.savefig('images/iter_{}.png'.format(epoch))
            plt.close()

            # Then plot the loss in a appealing format
            if epoch > 1: # Ensures there is data to plot
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
                axs[1].set_xlabel('Epoch')
                axs[1].set_ylabel('loss', fontsize=10)
                plt.savefig('images/'+'loss_tracking.png')
                plt.close()
        end = datetime.datetime.now()
        print('Epoch: {}/{},\t G loss: {:.4f}, D loss: {:.4f}\t| time: {}'.
              format(epoch, epochs, gloss, dloss, end-start))
