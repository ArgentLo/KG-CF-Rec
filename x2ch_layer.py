import tensorflow as tf
import numpy as np


# update "KG-Attentive Aggregation" after cleaning code
# Add "knowledge-embedded edges" after KG is obtained


class KG_Aggregator(object):
    """ 
    Aggregating Neigbors Messages through KG channel with attention on Edge Attributes
    """
    def __init__(self, name, input_dim, output_dim,
                 ffd_drop=0., attn_drop=0., usebias=False, act=tf.nn.elu):
        self.name = name
        self.attn_drop = attn_drop
        self.usebias = usebias
        self.act = act
        
        with tf.variable_scope(name) as scope:
            self.conv1 = tf.layers.Conv1D(filters=output_dim, kernel_size=1, name='conv1')
            self.conv2 = tf.layers.Conv1D(filters=1, kernel_size=1, name='conv2')
            if usebias:
                self.bias = tf.get_variable('bias',
                                            initializer=zero_init((output_dim)))

    def __call__(self, inputs):

        """
        1. Channel-wise Sigmoid of <Source, Edge>
        2. Integrate with Neighbor Embedding.
        3. Accumulate.
        """
        pass


#######################################################
#######################################################

    
class CF_Aggregator(object):
    """ 
    Aggregating Neigbors Messages through CF channel with attention
    """
    def __init__(self, name, input_dim, output_dim,
                 ffd_drop=0., attn_drop=0., usebias=False, act=tf.nn.elu):
        self.name = name
        self.attn_drop = attn_drop
        self.usebias = usebias
        self.act = act
        
        with tf.variable_scope(name) as scope:
            self.conv1 = tf.layers.Conv1D(filters=output_dim, kernel_size=1, name='conv1')
            self.conv2 = tf.layers.Conv1D(filters=1, kernel_size=1, name='conv2')
            if usebias:
                self.bias = tf.get_variable('bias',
                                            initializer=zero_init((output_dim)))
   
    def __call__(self, inputs):
        """
        Args:
            input: (self_vecs, neighbor_vecs, channel_vecs)
            self_vecs.shape = [batch_size, dim]
            neighbor_vecs.shape = [batch_size, num_samples, dim]
            channel_vecs.shape = [batch_size, num_samples, 1]
        """

        # Input
        self_vecs, neighbor_vecs, channel_vecs = inputs

        # 1. Reshape as [batch_size, 1, dim]
        # 2. Concatenate into [batch_size, 1+num_samples, dim]
        vecs = tf.concat([tf.expand_dims(self_vecs, axis=1), neighbor_vecs], axis=1)

        # REMOVE Dropout
        # # dropout
        # vecs = tf.nn.dropout(vecs, 1-self.ffd_drop)

        # transform and self attention
        with tf.variable_scope(self.name) as scope:

            # Inputs conversion
            vecs_trans = self.conv1(vecs) # [batch_size, 1+num_samples, output_dim]
            f_1 = self.conv2(vecs_trans)  # [batch_size, 1+num_samples, 1]
            f_2 = self.conv2(vecs_trans)


            #######################################################
            #######################################################

            logits = f_1 + tf.transpose(f_2, [0, 2, 1]) # [batch_size, 1+num_samples, 1+num_samples]
            coefs = tf.nn.softmax(tf.nn.leaky_relu(logits))

            # only maintain the target node for each batch
            coefs = tf.slice(coefs, [0,0,0], [-1,1,-1]) # [batch_size, 1, 1+num_samples]

            # channel (add one dim for self channel)
            self_channel = tf.slice(tf.ones_like(coefs), [0,0,0], [-1,1,1]) # [batch_size, 1, 1]
            channels = tf.concat((self_channel, channel_vecs), axis=1) # [batch_size, 1+num_samples, 1]
            channels = tf.transpose(channels, [0, 2, 1]) # [batch_size, 1, 1+num_samples]

            # Attention Apply: channel * attention
            coefs = tf.multiply(channels, coefs)

            #######################################################
            #######################################################

            # REMOVE Dropout
            # # dropout
            # coefs = tf.nn.dropout(coefs, 1-self.attn_drop)
            # vecs_trans = tf.nn.dropout(vecs_trans, 1-self.ffd_drop)

            # aggregate
            output = tf.matmul(coefs, vecs_trans) # [batch_size, 1, output_dim]
            output = tf.squeeze(output) # [batch_size, output_dim]

            # output
            if self.usebias:
                output += self.bias
        return self.act(output)    


def uniform_init(shape, scale=0.05):
    return tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)

def glorot_init(shape):
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    return tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)

def zeros_init(shape):
    return tf.zeros(shape, dtype=tf.float32)

def ones_init(shape):
    return tf.ones(shape, dtype=tf.float32)
