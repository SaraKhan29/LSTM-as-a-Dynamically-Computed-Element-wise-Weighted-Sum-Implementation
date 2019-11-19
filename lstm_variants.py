# -*- coding: utf-8 -*-
"""Module implementing variants of LSTM Cells."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import tensorflow as tf
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.ops.rnn_cell_impl import *


"""
Terminology for below LSTM cells
Input weights: W_z, W_i, W_f , W_o
Recurrent weights: R_z, R_i, R_f , R_o
Bias weights: b_z, b_i, b_f , b_o 
z is ctilde_t
y is h_t

"""



class LSTM_SRNNCell(rnn_cell.RNNCell):

  """
  This LSTM cell replaces S-RNN in the Content Layer with simple linear transformation
  Original Equation: z = g(tf.matmul(inputs, W_z) + tf.matmul(y_prev, R_z) + b_z) 
  New equation:  z = (tf.matmul(inputs, W_z)  + b_z)
  """
  def __init__(self, num_blocks, architecture= "LSTM"):
    self._num_blocks = num_blocks
    self._architecture = architecture
    self._state_is_tuple = True

  @property
  def input_size(self):
    return self._num_blocks

  @property
  def output_size(self):
    return self._num_blocks

  @property
  def state_size(self):
    return 2 * self._num_blocks

  def __call__(self, inputs, state, scope=None):
    with tf.variable_scope(scope or type(self).__name__):
        initializer = tf.random_uniform_initializer(-0.1, 0.1)

        def get_variable(name, shape):
              return tf.get_variable(name, shape, initializer=initializer, dtype=inputs.dtype)

        c_prev, y_prev = tf.split(state,num_or_size_splits=2,axis=1)

        W_z = get_variable("W_z", [self.input_size, self._num_blocks])
        W_i = get_variable("W_i", [self.input_size, self._num_blocks])
        W_f = get_variable("W_f", [self.input_size, self._num_blocks])
        W_o = get_variable("W_o", [self.input_size, self._num_blocks])

        R_z = get_variable("R_z", [self._num_blocks, self._num_blocks])
        R_i = get_variable("R_i", [self._num_blocks, self._num_blocks])
        R_f = get_variable("R_f", [self._num_blocks, self._num_blocks])
        R_o = get_variable("R_o", [self._num_blocks, self._num_blocks])

        b_z = get_variable("b_z", [1, self._num_blocks])
        b_i = get_variable("b_i", [1, self._num_blocks])
        b_f = get_variable("b_f", [1, self._num_blocks])
        b_o = get_variable("b_o", [1, self._num_blocks])

        g = h = tf.tanh
        z = (tf.matmul(inputs, W_z)  + b_z)
        i = tf.sigmoid(tf.matmul(inputs, W_i) + tf.matmul(y_prev, R_i)  + b_i)
        f = tf.sigmoid(tf.matmul(inputs, W_f) + tf.matmul(y_prev, R_f)  + b_f)
        c = tf.multiply(i, z) + tf.multiply(f, c_prev)
        o = tf.sigmoid(tf.matmul(inputs, W_o) + tf.matmul(y_prev, R_o) + b_o)
        y = tf.multiply(h(c), o)
        return y, tf.concat([c, y],axis=1)



class LSTM_GATESCell(rnn_cell.RNNCell):

  """
  This LSTM cell removes the output gate from LSTM
  """
  def __init__(self, num_blocks, architecture= "LSTM_GATES"):
    self._num_blocks = num_blocks

  @property
  def input_size(self):
    return self._num_blocks

  @property
  def output_size(self):
    return self._num_blocks

  @property
  def state_size(self):
    return 2 * self._num_blocks

  def __call__(self, inputs, state, scope=None):
    with tf.variable_scope(scope or type(self).__name__):
      initializer = tf.random_uniform_initializer(-0.1, 0.1)
      initializer1 = tf.constant_initializer(0.0)

      def get_variable(name, shape):
        return tf.get_variable(name, shape, initializer=initializer, dtype=inputs.dtype)

      #to keep gates zeros. Only LSTM without gates
      def get_variable1(name, shape):
        return tf.get_variable(name, shape, initializer= initializer1, dtype=inputs.dtype)


      c_prev, y_prev = tf.split(state,num_or_size_splits=2,axis=1)

      W_z = get_variable("W_z", [self.input_size, self._num_blocks])
      W_i = get_variable1("W_i", [self.input_size, self._num_blocks])
      W_f = get_variable1("W_f", [self.input_size, self._num_blocks])
      W_o = get_variable1("W_o", [self.input_size, self._num_blocks])

      R_z = get_variable("R_z", [self._num_blocks, self._num_blocks])
      R_i = get_variable1("R_i", [self._num_blocks, self._num_blocks])
      R_f = get_variable1("R_f", [self._num_blocks, self._num_blocks])
      R_o = get_variable1("R_o", [self._num_blocks, self._num_blocks])

      b_z = get_variable("b_z", [1, self._num_blocks])
      b_i = get_variable1("b_i", [1, self._num_blocks])
      b_f = get_variable1("b_f", [1, self._num_blocks])
      b_o = get_variable1("b_o", [1, self._num_blocks])

     
      # Values of i,f,o are zeros
      g = h = tf.tanh
      z = g(tf.matmul(inputs, W_z) + tf.matmul(y_prev, R_z) + b_z)
      i = (tf.matmul(inputs, W_i) + tf.matmul(y_prev, R_i)  + b_i)
      f = (tf.matmul(inputs, W_f) + tf.matmul(y_prev, R_f)  + b_f)
      c = tf.multiply(i, z) + tf.multiply(f, c_prev)
      o =  (tf.matmul(inputs, W_o) + tf.matmul(y_prev, R_o) + b_o)
      y = tf.multiply(h(c), o)
      return y, tf.concat([c, y],axis=1)


class LSTM_SRNN_OUTCell(rnn_cell.RNNCell):
  """
    original equations:
            z = g(tf.matmul(inputs, W_z) + tf.matmul(y_prev, R_z) + b_z)
            o = tf.sigmoid(tf.matmul(inputs, W_o) + tf.matmul(y_prev, R_o) + b_o)
            y = tf.multiply(h(c), o)
    New equations:
            z = (W_z + b_z)
            o is removed         
            y=h(c)  
  """



  def __init__(self, num_blocks, architecture="LSTM_SRNN_OUT"):
      self._num_blocks = num_blocks

  @property
  def input_size(self):
    return self._num_blocks

  @property
  def output_size(self):
    return self._num_blocks

  @property
  def state_size(self):
    return 2 * self._num_blocks

  def __call__(self, inputs, state, scope=None):
    with tf.variable_scope(scope or type(self).__name__):
      initializer = tf.random_uniform_initializer(-0.1, 0.1)

      def get_variable(name, shape):
        return tf.get_variable(name, shape, initializer=initializer, dtype=inputs.dtype)

      c_prev, y_prev = tf.split(state,num_or_size_splits=2,axis=1)

      W_z = get_variable("W_z", [self.input_size, self._num_blocks])
      W_i = get_variable("W_i", [self.input_size, self._num_blocks])
      W_f = get_variable("W_f", [self.input_size, self._num_blocks])
      W_o = get_variable("W_o", [self.input_size, self._num_blocks])

      R_z = get_variable("R_z", [self._num_blocks, self._num_blocks])
      R_i = get_variable("R_i", [self._num_blocks, self._num_blocks])
      R_f = get_variable("R_f", [self._num_blocks, self._num_blocks])
      R_o = get_variable("R_o", [self._num_blocks, self._num_blocks])

      b_z = get_variable("b_z", [1, self._num_blocks])
      b_i = get_variable("b_i", [1, self._num_blocks])
      b_f = get_variable("b_f", [1, self._num_blocks])
      b_o = get_variable("b_o", [1, self._num_blocks])

      g = h = tf.tanh
      z = (tf.matmul(inputs, W_z)  + b_z)
      i = tf.sigmoid(tf.matmul(inputs, W_i) + tf.matmul(y_prev, R_i)  + b_i)
      f = tf.sigmoid(tf.matmul(inputs, W_f) + tf.matmul(y_prev, R_f)  + b_f)
      c = tf.multiply(i, z) + tf.multiply(f, c_prev)
      #o = tf.sigmoid(tf.matmul(inputs, W_o) + tf.matmul(y_prev, R_o) + b_o)
      y = h(c)
      return y, tf.concat([c, y],axis=1)




class LSTM_SRNN_HIDDENCell(rnn_cell.RNNCell):

  """
  This LSTM cell replaces S-RNN in the Content Layer with simple linear transformation
  Original Equation: z = g(tf.matmul(inputs, W_z) + tf.matmul(y_prev, R_z) + b_z) 
  New equation:  z = (tf.matmul(inputs, W_z)  + b_z)
  """
  def __init__(self, num_blocks, architecture= "LSTM_SRNN_HIDDEN"):
    self._num_blocks = num_blocks
    self._architecture = architecture

  @property
  def input_size(self):
    return self._num_blocks

  @property
  def output_size(self):
    return self._num_blocks

  @property
  def state_size(self):
    return 2 * self._num_blocks

  def __call__(self, inputs, state, scope=None):
      with tf.variable_scope(scope or type(self).__name__):
          initializer = tf.random_uniform_initializer(-0.1, 0.1)

          def get_variable(name, shape):
                return tf.get_variable(name, shape, initializer=initializer, dtype=inputs.dtype)

          c_prev, y_prev = tf.split(state,num_or_size_splits=2,axis=1)

          W_z = get_variable("W_z", [self.input_size, self._num_blocks])
          W_i = get_variable("W_i", [self.input_size, self._num_blocks])
          W_f = get_variable("W_f", [self.input_size, self._num_blocks])
          W_o = get_variable("W_o", [self.input_size, self._num_blocks])

          R_z = get_variable("R_z", [self._num_blocks, self._num_blocks])
          R_i = get_variable("R_i", [self._num_blocks, self._num_blocks])
          R_f = get_variable("R_f", [self._num_blocks, self._num_blocks])
          R_o = get_variable("R_o", [self._num_blocks, self._num_blocks])

          b_z = get_variable("b_z", [1, self._num_blocks])
          b_i = get_variable("b_i", [1, self._num_blocks])
          b_f = get_variable("b_f", [1, self._num_blocks])
          b_o = get_variable("b_o", [1, self._num_blocks])

          g = h = tf.tanh
          z = (tf.matmul(inputs, W_z)  + b_z)
          i = tf.sigmoid(tf.matmul(inputs, W_i)  + b_i)
          f = tf.sigmoid(tf.matmul(inputs, W_f) + b_f)
          c = tf.multiply(i, z) + tf.multiply(f, c_prev)
          o = tf.sigmoid(tf.matmul(inputs, W_o)  + b_o)
          y = tf.multiply(h(c), o)
          return y, tf.concat([c, y],axis=1)


