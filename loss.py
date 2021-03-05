import math
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers, optimizers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Embedding, Input, Lambda, Add
from tensorflow.python.keras.models import Model
from tensorflow.keras import regularizers

from utils import *

# Bayesian Personalized Ranking Loss function for X-2ch training
# (negative example ratio == 1)


def line_loss(y_true, y_pred):

#     y_true = K.print_tensor(y_true, message="y_true is: ")
#     y_pred = K.print_tensor(y_pred, message="y_pred is: ")
    
    r1 = y_true * y_pred
    r2 = K.sigmoid(r1)
    r3 = K.log(r2)
    result = - K.mean(r3)
    
    return result

def affinity(y_ture, y_pred):
    return tf.reduce_sum(y_ture * y_pred, axis=1)

def neg_cost(inputs, neg_samples):
    return tf.matmul(inputs, tf.transpose(neg_samples))

def skipgram_loss(y_ture, y_pred, neg_samples):
    aff = affinity(y_ture, y_pred)
    neg_aff = neg_cost(y_ture, neg_samples)
    neg_cost = tf.log(tf.reduce_sum(tf.exp(neg_aff), axis=1))
    return tf.reduce_sum(aff - neg_cost)

def xent_loss(y_ture, y_pred, neg_samples, neg_sample_weight=1.0):
    aff = affinity(y_ture, y_pred)
    neg_aff = neg_cost(y_ture, neg_samples)
    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(aff), logits=aff)
    neg_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_aff), logits=neg_aff)
    return tf.reduce_sum(true_xent) + neg_sample_weight * (tf.reduce_mean(neg_xent))
