import tensorflow as tf

def get_shape(tensor): # static shape
    return tensor.get_shape().as_list()

def batch_normalization(*args, **kwargs):
    with tf.name_scope('bn'):
        bn = tf.layers.batch_normalization(*args, **kwargs)
    return bn


def make_gif(path='images/'):
    import imageio
    import os
    images = []
    processed_names = []
    filenames = os.listdir(path)
    for filename in filenames:
        if filename.startswith('iter'):
            processed_names.append(filename)
    print(processed_names[0])
    print(processed_names[0][5:-4])
    processed_names = sorted(processed_names, key = lambda s: int(s[5:-4]))
    print(processed_names)
    for filename in processed_names:
        images.append(imageio.imread(path+filename))
    imageio.mimsave('images/training.gif', images, duration=0.1)


if __name__ == '__main__':
    make_gif()
