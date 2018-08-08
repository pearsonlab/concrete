import os
import tensorflow as tf
import numpy as np
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

np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)

def run_model(FLAGS):
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    obs = tf.placeholder(tf.float32, [None, FLAGS.obs_dim], "observations")
    data = np.load(FLAGS.train_data_dir)

    t = tf.get_variable(
        "iteration", dtype=tf.int32, initializer=0, trainable=False)

    with tf.name_scope("temperature"):
        # temp_gen = tf.get_variable(
        #     "t_gen", dtype=tf.float32,
        #     initializer=FLAGS.init_temp, trainable=False)
        # temp_rec = tf.get_variable(
        #     "t_rec", dtype=tf.float32,
        #     initializer=FLAGS.init_temp, trainable=False)
        temp_gen = tf.train.exponential_decay(
            FLAGS.init_temp, t, 100, 0.9, staircase=True, name="t_gen")
        temp_rec = tf.train.exponential_decay(
            FLAGS.init_temp, t, 100, 0.9, staircase=True, name="t_rec")
        tf.summary.scalar("generative", temp_gen)
        tf.summary.scalar("recognition", temp_rec)

    params = get_model_params(
        FLAGS.n_states, FLAGS.obs_dim, FLAGS.n_layers, FLAGS.layer_dim,
        temp_gen, temp_rec)

    gen = HMM_gen(obs, params)
    rec = HMM_rec(obs, params)

    with tf.name_scope("samples"):
        pred_probs = tf.nn.softmax(rec.sample(), name="predicted_probability")
        gen_sample = tf.nn.softmax(gen.sample(), name="generated_sample")

    with tf.name_scope("KLqp"):
        q_sample = tf.identity(rec.sample(), "posterior_sample")
        loss = tf.subtract(rec.log_prob(q_sample), gen.log_prob(q_sample),
                           "loss")

        tf.summary.scalar("estimated_ELBO", -loss)
        tf.summary.scalar("p_log_prob", gen.log_prob(q_sample))
        tf.summary.scalar("q_log_prob", rec.log_prob(q_sample))
        summary_op = tf.summary.merge_all()

        train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(
            loss, t, gen.var + rec.var, name="train_op")

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(
            FLAGS.model_dir + "/log", tf.get_default_graph())
        tf.global_variables_initializer().run()
        sess_saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=FLAGS.max_ckpt,
            name="session_saver")

        for i in range(FLAGS.n_epochs):
            if i == 0 or (i + 1) % 100 == 0:
                print("Entering epoch {} ...".format(i + 1))

            _, summary = sess.run(
                [train_op, summary_op], {obs: data})
            train_writer.add_summary(summary, i)
            train_writer.flush()

            if (i + 1) % FLAGS.freq_ckpt == 0:
                sess_saver.save(
                    sess, FLAGS.model_dir + "/saved_model",
                    global_step=(i + 1), latest_filename="ckpt")
                print("Model saved after {} epochs.".format(i + 1))

        train_writer.close()
        print("Training completed.")


def main(_):
    run_model(FLAGS)


if __name__ == "__main__":
    tf.app.run()
