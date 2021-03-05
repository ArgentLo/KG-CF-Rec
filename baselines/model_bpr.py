import tensorflow as tf
import os


#######################################################
# Construct BPR by take dot product of User-Item Embedding
# --> Only CF signals is supported (KG ignored in BPR)
#######################################################


class BPR(object):

    def __init__(self, data_config, pretrain_data, args):

        # Number to users/items
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        # Hyper-Parameters
        self.lr = args.lr
        self.regs = eval(args.regs)
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        # Placeholder definition
        self.users = tf.placeholder(tf.int32, shape=[None,], name='users')
        self.pos_items = tf.placeholder(tf.int32, shape=[None,], name='pos_items')
        self.neg_items = tf.placeholder(tf.int32, shape=[None,], name='neg_items')

        # Variable definition
        self.weights = self._init_weights()

        # Original embedding.
        u_e = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        pos_i_e = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        neg_i_e = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)


        def _create_bpr_loss(self, users, pos_items, neg_items):
            pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
            neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

            regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)

            maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))

            mf_loss = tf.negative(tf.reduce_mean(maxi))
            reg_loss = self.regs[0] * regularizer

            return mf_loss, reg_loss


        # Optimization process.
        self.base_loss, self.reg_loss = self._create_bpr_loss(u_e, pos_i_e, neg_i_e)
        self.kge_loss = tf.constant(0.0, tf.float32, [1])
        self.loss = self.base_loss + self.kge_loss + self.reg_loss

        # self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.loss)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)


        # For eval: predictions for all users.
        self.batch_predictions = tf.matmul(u_e, pos_i_e, transpose_a=False, transpose_b=True)
        self._statistics_params()


    def eval(self, sess, feed_dict):
        batch_predictions = sess.run(self.batch_predictions, feed_dict)
        return batch_predictions


    def _statistics_params(self):
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)


    # Run training in tf.compat.v1.Session
    def train(self, sess, feed_dict):

        return sess.run([self.opt, 
                         self.loss, 
                         self.base_loss, 
                         self.kge_loss, 
                         self.reg_loss], feed_dict)


    # If new training --> initialize weight of all layers (user/item embeddings)
    def _init_weights(self):

        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()

        if self.pretrain_data not None:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                    name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                    name='item_embedding', dtype=tf.float32)
            print('using pretrained initialization')

        else:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
            print('using xavier initialization')

        return all_weights

