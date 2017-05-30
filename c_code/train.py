"""
    Train model
"""
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from model import Model
from logger import Logger

log = Logger("Train")

class Train(Model):
    """
        Train the model to write c code by itslef !
    """

    def __init__(self):
        Model.__init__(self, training=True)

        log.info("Init batches...")
        self.init_dataset()
        log.info("Batches are ready.")

    def init_dataset(self):
        """
            Init the dataset and store it inside the DataContainer class
        """
        self.batch_inputs = []
        self.batch_targets = []
        encoded = self.encoded

        batch_size = self.BATCH_SIZE * self.SEQUENCE_SIZE
        n_batches = len(self.encoded) // batch_size
        encoded = encoded[:n_batches * batch_size]
        encoded = encoded.reshape((self.BATCH_SIZE, -1))

        for n in range(0, encoded.shape[1], self.SEQUENCE_SIZE):

            x = encoded[:, n:n+self.SEQUENCE_SIZE]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]

            self.batch_inputs.append(x)
            self.batch_targets.append(y)

    def train_model(self, epochs):
        """
            Train the model

            **input : **
                *epochs (int) Numbers of epoch
        """
        for e in range(epochs):

            # Train network
            n_state = self.get_new_state()

            for inputs, targets in zip(self.batch_inputs, self.batch_targets):
                t = time.time()
                batch_loss, n_state, _ = self.optimize_model(inputs, targets, n_state)
                log.info("Epoch : %s Training loss : %2f, sec/batch : %2f" % (e, batch_loss, (time.time() - t)))

            # We save the model at the ned of each epoch
            log.info("Save model...")
            self.save_model()

if __name__ == '__main__':
    tr = Train()
    tr.train_model(1000000)
