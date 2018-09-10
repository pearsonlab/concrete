import os
import tensorflow as tf
import numpy as np
from concrete import get_HMM_params, HMM_gen, HMM_rec
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


MODEL_DIR = "HMM_concrete"
DATA_DIR = ""
NUM_STATES = 2
OBS_DIM = 2
GEN_LAYER_SIZES = "32,32"
REC_LAYER_SIZES = "32,32"
INIT_TEMPERATURE = 1e-3
SEED = 1234
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
MAX_CKPT = 10
FREQ_CKPT = 5


flags = tf.app.flags

flags.DEFINE_string("model_dir", MODEL_DIR,
                    "Directory where the model is saved")
flags.DEFINE_string("data_dir", DATA_DIR, "Directory of training data file")

flags.DEFINE_integer("num_states", NUM_STATES, "Number of states")
flags.DEFINE_integer("obs_dim", OBS_DIM, "Dimension of observations")
flags.DEFINE_string("gen_layer_sizes", GEN_LAYER_SIZES,
                    "Dimension of densely-connected layer output in \
                    transition neural network (generative), separated by ,")
flags.DEFINE_string("rec_layer_sizes", REC_LAYER_SIZES,
                    "Dimension of densely-connected layer output in \
                    transition neural network (recognition), separated by ;")
flags.DEFINE_float("init_temp", INIT_TEMPERATURE,
                   "Initial temperature of concrete distribution(s)")

flags.DEFINE_integer("seed", SEED, "Random seed of computational graph")
flags.DEFINE_float("lr", LEARNING_RATE, "Initial learning rate")
flags.DEFINE_integer("num_epochs", NUM_EPOCHS, "Number of iterations \
                     algorithm runs through the data set")
flags.DEFINE_integer("max_ckpt", MAX_CKPT, "Maximum number of checkpoints \
                     to keep in the model directory")
flags.DEFINE_integer("freq_ckpt", FREQ_CKPT, "Frequency of saving \
                     checkpoints to the model directory")

FLAGS = flags.FLAGS

np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)

def run_model(FLAGS):
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    global_step = tf.train.get_or_create_global_step()
    with tf.name_scope("temperature"):
        # t_gen = tf.get_variable(
        #     "gen", dtype=tf.float32,
        #     initializer=FLAGS.init_temp, trainable=False)
        # t_rec = tf.get_variable(
        #     "rec", dtype=tf.float32,
        #     initializer=FLAGS.init_temp, trainable=False)
        t_gen = tf.train.exponential_decay(
            FLAGS.init_temp, global_step, 100, 0.9,
            staircase=True, name="gen")
        t_rec = tf.train.exponential_decay(
            FLAGS.init_temp, global_step, 100, 0.9,
            staircase=True, name="rec")
        tf.summary.scalar("generative", t_gen)
        tf.summary.scalar("recognition", t_rec)

    obs = tf.placeholder(tf.float32, [None, FLAGS.obs_dim], "observations")
    data = np.load(FLAGS.data_dir)

    gen_layer_sizes = [int(n) for n in FLAGS.gen_layer_sizes.split(",")]
    rec_layer_sizes = [int(n) for n in FLAGS.rec_layer_sizes.split(",")]
    params = get_HMM_params(
        FLAGS.num_states, FLAGS.obs_dim, gen_layer_sizes, rec_layer_sizes,
        t_gen, t_rec)

    gen = HMM_gen(params)
    rec = HMM_rec(obs, params)

    with tf.name_scope("samples"):
        ret_q = rec.sample_states(len(data))
        pred_states = tf.exp(ret_q[0], "predicted_states")
        pred_logits = tf.identity(ret_q[1], "posterior_logits")

        T = tf.placeholder_with_default(len(data), [], "length")
        ret_p = gen.sample_states(T)
        gen_states = tf.exp(ret_p[0], "generated_states")
        gen_logits = tf.identity(ret_p[1], "generated_logits")
        gen_data = tf.identity(gen.sample_data(T, gen_states),
                               "generated_data")

    with tf.name_scope("VI"):
        q_sample = tf.identity(rec.sample_states(len(data))[0],
                               "posterior_sample")
        log_likelihood_data = tf.identity(
            gen.log_likelihood_data(obs, q_sample), "log_likelihood_data")
        kl_divergence = tf.subtract(
            gen.log_prob_states(q_sample), rec.log_prob_states(q_sample),
            "kl_divergence")
        elbo = tf.add(log_likelihood_data, kl_divergence, "ELBO")

        tf.summary.scalar("Estimated_ELBO", elbo)
        tf.summary.scalar("Log_Likelihood", log_likelihood_data)
        tf.summary.scalar("KLpq", kl_divergence)
        train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(
            -elbo, global_step, gen.var + rec.var, name="train_op")

    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(
            FLAGS.model_dir + "/log", tf.get_default_graph())
        tf.global_variables_initializer().run()
        sess_saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=FLAGS.max_ckpt,
            name="session_saver")

        for i in range(FLAGS.num_epochs):
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
