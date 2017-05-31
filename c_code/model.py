# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import numpy as np
import json
from logger import Logger

log = Logger("Model")

class Model(object):
    """
        Class use to load the dataset and to build
        tensorflow Graph.
    """

    DATASET_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    DATASET_FILE = os.path.join(DATASET_FOLDER, "dataset.dt")
    CHECKPOINT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    TENSORBOARD_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs/train")

    SEQUENCE_SIZE = 100
    BATCH_SIZE = 200

    HIDDEN_LAYER_SIZE = 512
    LSTM_SIZE = 2

    LR = 0.0005

    KEEP_PROB = 0.5

    def __init__(self, training=False, ckpt=None):
        """
            **input : **
                *training (Boolean)
                *ckpt (str) Path to the ckpt file to use
        """
        super(Model, self).__init__()

        # Create the checkpoints folder used to save the session
        if not os.path.exists(self.CHECKPOINT_FOLDER):
            os.makedirs(self.CHECKPOINT_FOLDER)

        # Loading the dataset
        log.info("Loading dataset ...")
        self.encoded, self.vocab_to_int, self.int_to_vocab = Model.get_dataset()
        log.info("Dataset loaded.")

        # io_size is use know the numbers of possible input, and the numbers of
        # output neurons of our model.
        self.io_size = len(self.vocab_to_int)

        # This model have to be built to be trained or only to be used.
        if training == True:
            self.sequence_size = self.SEQUENCE_SIZE
            self.batch_size = self.BATCH_SIZE
        else:
            self.sequence_size = 1
            self.batch_size = 1

        # Build the Graph
        self.inputs, self.targets, self.keep_prob = self.build_inputs() # Placeholders
        cell_output, self.initial_state, self.final_state = self.build_lstm(self.inputs, self.keep_prob) # LSTM
        self.softmax, logits = self.build_output(cell_output) # LSTM output
        self.loss = self.build_loss(logits, self.targets) # Cost function
        self.optimizer = self.build_optimizer(self.loss) # Optimizer

        # Build session
        self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.session = tf.Session()

        self.session.run(init)
        # Tensorboard
        self.tensorboard = tf.summary.merge_all()
        model_param = "bs_%s_hs_%s_stem_size_%s_lr_%s" % (self.BATCH_SIZE, self.HIDDEN_LAYER_SIZE, self.LSTM_SIZE, self.LR)
        self.train_writer = tf.summary.FileWriter(os.path.join(self.TENSORBOARD_LOG, model_param), self.session.graph)
        self.train_writer_it = 0

        self.model_it = 0

        if ckpt is not None:
            self.saver.restore(self.session, ckpt)
            log.info("checkpoints succesfully loaded")

    def get_new_state(self):
        """
            ** return (Tensor) **
        """
        return self.session.run(self.initial_state)

    def optimize_model(self, x, y, state):
        """
            Optimize the model and return the current loss value of this batch
            and the final cell state of our RNN
            **input : **
                *x (Numpy array)
                *y (Numpy array)
        """
        summary, batch_loss, n_state, _ = self.session.run([self.tensorboard, self.loss, self.final_state, self.optimizer], feed_dict={self.inputs: x, self.targets: y, self.keep_prob: self.KEEP_PROB, self.initial_state: state})
        self.train_writer.add_summary(summary, self.train_writer_it)
        self.train_writer_it += 1
        return batch_loss, n_state, _

    def save_model(self):
        """
            Save the model
        """
        self.model_it += 1
        print ("Save model")
        self.saver.save(self.session, "checkpoints/dump_model_%s.ckpt" % self.model_it)
        print ('Model succesfully saved !')

    @staticmethod
    def get_dataset():
        """
            Method used to build/retrive the dataset
        """

        if os.path.isfile(Model.DATASET_FILE):
            with open(Model.DATASET_FILE, "r") as d:
                d = json.loads(d.read())
                return np.array(d["encoded"]), d["vocab_to_int"], d["int_to_vocab"]

        # List of file in the dataset directory
        all_file = os.listdir(Model.DATASET_FOLDER)
        # Filter : Select all c file
        all_file_name = np.array([f for f in all_file if f.find(".c") != -1])

        content = ""
        for name in all_file_name:
            with open(os.path.join(Model.DATASET_FOLDER, name), "r") as f:
                content += f.read() + "\n"

        # Convert the string into a list of interger
        vocab = set(content)
        vocab_to_int = {c: i for i, c in enumerate(vocab)}
        int_to_vocab = dict(enumerate(vocab))
        encoded = np.array([vocab_to_int[c] for c in content], dtype=np.int32)

        with open(Model.DATASET_FILE, "w+") as d:
            json_file = json.dumps({"encoded" : [int(i) for i in encoded], "vocab_to_int" : vocab_to_int, "int_to_vocab" : int_to_vocab})
            d.write(json_file)
            d.close()

        return encoded, vocab_to_int, int_to_vocab

    def build_lstm(self, inputs, keep_prob):
        """
            Build our RNN model

            **input : **
                *inputs (tf.Placeholder)
                *keep_prob (tf.Placeholder)
            **return (Tuple (Tensor operation, Tensor operation, Tensor operation)) **
                *cell_outputs : Outputs value of each LSTM cell
                *initial_state : LSTM cell with all neurons set to zero.
                *final_state : State values of the last cell.
        """
        with tf.name_scope("LSTM"):
            def create_cell():
                lstm = tf.contrib.rnn.BasicLSTMCell(self.HIDDEN_LAYER_SIZE)
                drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
                return drop

            cell = tf.contrib.rnn.MultiRNNCell([create_cell() for _ in range(self.LSTM_SIZE)])
            initial_state = cell.zero_state(self.batch_size, tf.float32)

            x_one_hot = tf.one_hot(inputs, self.io_size)
            cell_outputs, final_state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state)

        return cell_outputs, initial_state, final_state

    def build_output(self, cell_outputs):
        """
            Create an output layer at the top of our RNN model

            **input : **
                *cell_outputs (Tensor operation) Outputs value of each LSTM cell
            **return (Tuple (Tensor operation, Tensor operation))
                *softmax
                *logits
        """
        with tf.name_scope("graph_outputs"):
            #seq_output = tf.concat(cell_outputs, axis=1, name="concat_all_cell_outputs")
            x = tf.reshape(cell_outputs, [-1, self.HIDDEN_LAYER_SIZE], name="reshape_x")

            with tf.name_scope('output_layer'):
                w = tf.Variable(tf.truncated_normal((self.HIDDEN_LAYER_SIZE, self.io_size), stddev=0.1), name="weights")
                b = tf.Variable(tf.zeros(self.io_size), name="bias")
                tf.summary.histogram("weights", w)
                tf.summary.histogram("bias", b)

            logits = tf.add(tf.matmul(x , w), b, name= "logits")
            softmax = tf.nn.softmax(logits, name='predictions')
            tf.summary.histogram("softmax", softmax)

        return softmax, logits

    def build_loss(self, logits, targets):
        """
            We use to measure our error : softmax_cross_entropy_with_logits

            **input : **
                *logits (Tensor operation)
                *targets (tf.Placeholder)
            ** return (Tensor operation) **
                *loss : compute the loss of the model
        """
        with tf.name_scope("cost"):
            y_one_hot = tf.one_hot(targets, self.io_size, name="y_to_one_hot")
            y_reshaped = tf.reshape(y_one_hot, logits.get_shape(), name="reshape_one_hot")
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
            loss = tf.reduce_mean(loss)
            tf.summary.scalar('loss', loss)

        return loss

    def build_optimizer(self, loss):
        """
            Apply optimizer to the loss
            We choose for this model AdamOptimizer

            **input : **
                *loss (Tensor operation) compute the loss of the model
            **return (Tensor operation) **
                *optimizer : to into the session
        """
        with tf.name_scope("train"):
            adam = tf.train.AdamOptimizer(self.LR)
            optimizer = adam.minimize(loss)

        return optimizer

    def build_inputs(self):
        """
            Build all tensorflow placeholder

            ** return (Tuple (tf.Placeholder, tf.Placeholder, tf.Placeholder))
                *Inputs placeholder
                *Targets placeholder
                *Probability to keep neurons during dropout
        """
        with tf.name_scope("graph_inputs"):
            inputs = tf.placeholder(tf.int32, [None, self.sequence_size], name='placeholder_inputs')
            targets = tf.placeholder(tf.int32, [None, self.sequence_size], name='placeholder_targets')
            keep_prob = tf.placeholder(tf.float32, name='placeholder_keep_prob')

        return inputs, targets, keep_prob
