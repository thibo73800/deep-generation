# -*- coding: utf-8 -*-
"""
Genearte a new c file.

Usage:
  use.py <ckpt>

Options:
  -h --help                 Show this help.
  <ckpt>                    Path to the ckpt file to use
"""

import tensorflow as tf
import numpy as np
from model import Model
from docopt import docopt

class Use(Model):
    """
        Class use to generate a new c file
    """

    def __init__(self, ckpt):
        """
            **input : **
                **Path to the ckpt file to use
        """
        Model.__init__(self, training=False, ckpt=ckpt)

        self.code = [c for c in "\n\n static int"]

    def add_to_code(self, softmax):
        """
            Add a new letter to the code based on the softmax value
            **input : **
                *softmax (numpy.ndarray)
            **return (Int)
                Return softmax index choosen
        """
        softmax = np.squeeze(softmax)
        softmax[np.argsort(softmax)[:-4]] = 0
        softmax = softmax / np.sum(softmax)
        c = np.random.choice(len(self.int_to_vocab), 1, p=softmax)[0]
        self.code.append(self.int_to_vocab[str(c)])
        return c

    def create_code(self, size):
        """
            Generate a new piece of code
            **input : **
                *size (Integer) Numbers of code charactere to generate
            **return (str) **
        """

        previous_state = self.session.run(self.initial_state)

        for c in self.code:
            x = np.array([[self.vocab_to_int[c]]])
            feed = {self.inputs: x, self.keep_prob: 1., self.initial_state: previous_state}
            softmax, previous_state = self.session.run([self.softmax, self.final_state], feed_dict=feed)

        c = self.add_to_code(softmax)

        for i in range(size):
            x[0,0] = c
            feed = {self.inputs: x, self.keep_prob: 1., self.initial_state: previous_state}
            softmax, previous_state = self.session.run([self.softmax, self.final_state], feed_dict=feed)
            c = self.add_to_code(softmax)

        return ''.join(self.code)

if __name__ == '__main__':
    arguments = docopt(__doc__)
    use_model = Use(arguments["<ckpt>"])
    code = use_model.create_code(4000)

    with open("c_file.c", "w+") as s:
        s.write(code)
        s.close()
