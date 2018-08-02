import os
import tensorflow as tf
import numpy as np
import edward as ed
from concrete import get_model_params, HMM_gen, HMM_rec


MODEL_DIR = "HMM_concrete"
TRAINING_DATA_DIR = ""
N_STATES = 2
OBS_DIM = 2
N_LAYERS = 3
LAYER_DIM = 32
INIT_TEMP = 1e-3
SEED = 1234
LEARNING_RATE = 1e-3
N_EPOCHS = 100
MAX_CKPT = 10
FREQ_CKPT = 5


flags = tf.app.flags

flags.DEFINE_string("model_dir", MODEL_DIR,
                    "Directory where the model is saved")
flags.DEFINE_string("train_data_dir", TRAINING_DATA_DIR,
                    "Directory of training data file")

flags.DEFINE_integer("n_states", N_STATES, "Number of states")
flags.DEFINE_integer("obs_dim", OBS_DIM, "Dimension of observations")
flags.DEFINE_integer("n_layers", N_LAYERS,
	                 "Number of layers in neural network (recognition)")
flags.DEFINE_integer("layer_dim", LAYER_DIM, "Number of hidden units \
	                 in each dense layer of neural network (recognition)")
flags.DEFINE_float("init_temp", INIT_TEMP,
	               "Initial temperature of concrete distribution(s)")

flags.DEFINE_integer("seed", SEED, "Random seed")
flags.DEFINE_float("lr", LEARNING_RATE, "Initial learning rate")
flags.DEFINE_integer("n_epochs", N_EPOCHS, "Number of iterations algorithm \
                    runs through the training set")
flags.DEFINE_integer("max_ckpt", MAX_CKPT,
                     "Maximum number of checkpoints to keep in the directory")
flags.DEFINE_integer("freq_ckpt", FREQ_CKPT, "Frequency of saving \
                     checkpoints to the directory")

FLAGS = flags.FLAGS

ed.set_seed(FLAGS.seed)

def run_model(FLAGS):
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    obs = tf.placeholder(tf.float32, [None, FLAGS.obs_dim], "observations")
    data = np.load(FLAGS.train_data_dir)

    temp_gen = tf.get_variable(
        "temperature_gen", dtype=tf.float32,
        initializer=FLAGS.init_temp, trainable=False)
    temp_rec = tf.get_variable(
        "temperature_rec", dtype=tf.float32,
        initializer=FLAGS.init_temp, trainable=False)

    params = get_model_params(
        FLAGS.n_states, FLAGS.obs_dim, FLAGS.n_layers, FLAGS.layer_dim,
        temp_gen, temp_rec)

    gen = HMM_gen(obs, params)
    rec = HMM_rec(obs, params)

    pred_probs = tf.nn.softmax(rec.sample(), name="predicted_probability")
    new_sample = tf.nn.softmax(gen.sample(), name="sample")

    inference = ed.KLqp({gen: rec})
    optimizer = tf.train.AdamOptimizer(FLAGS.lr)
    inference.initialize(
        var_list=(gen.var + rec.var), optimizer=optimizer,
        logdir=FLAGS.model_dir + "/log")

    sess = ed.get_session()
    tf.global_variables_initializer().run()
    sess_saver = tf.train.Saver(
    	tf.global_variables(), max_to_keep=FLAGS.max_ckpt,
        name="session_saver")

    for i in range(FLAGS.n_epochs):
        if i == 0 or (i + 1) % 10 == 0:
            print("Entering epoch {} ...".format(i + 1))

        inference.update({obs: data})

        if (i + 1) % FLAGS.freq_ckpt == 0:
            sess_saver.save(
            	sess, FLAGS.model_dir + "/saved_model",
                global_step=(i + 1), latest_filename="ckpt")
            print("Model saved after {} epochs.".format(i + 1))

    inference.finalize()
    print("Training completed.")


def main(_):
    run_model(FLAGS)


if __name__ == "__main__":
    tf.app.run()
