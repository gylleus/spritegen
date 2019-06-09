from generator import Generator
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import shutil

#from mlagents.trainers import tensorflow_to_barracuda as tf2bc

def save_generator(weights_dir, out_path="out"):
    shutil.rmtree(out_path, True)
    builder = tf.saved_model.builder.SavedModelBuilder(out_path)
    final_model_path = out_path + '/frozen_graph.bytes'

    with tf.Session() as sess:
        # Ugly solution to load graph of Generator
        gen = Generator()
        zass = tf.random_normal(shape=(64, 100), dtype='float32')
        dafuq_data = tf.stop_gradient(gen(zass))
        gen.load_weights(weights_dir)
        builder.add_meta_graph_and_variables(
                sess=sess,
                tags=[tf.saved_model.tag_constants.SERVING]
        )
        builder.save()
        checkpoint_path = out_path + "/generator.chkp"
        tf.train.Saver().save(sess=sess, save_path=checkpoint_path)
        freeze_graph.freeze_graph(out_path + "/saved_model.pb", None, True,
                              out_path + '/generator.chkp', "generator/output_node",
                              "save/restore_all", "save/Const:0",
                              final_model_path, True, "",
                              input_saved_model_dir=out_path)

        # Tensorflow to barracuda does not work for this model for some reason
        # tf2bc.convert(out_path + '/frozen_graph.bytes', out_path + '/brain.nn')
        # logger.info('Exported ' + self.model_path + '.nn file')
        print('Exported model to ' + final_model_path + ' successfully!')