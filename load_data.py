import collections
import numpy as np
import tensorflow as tf
import random as rd
from utils import *


#######################################################
###########    load x-2ch CF and KG data    ###########
#######################################################

class Dataloader(object):

    def __init__(self, args, path):

        self.args = args
        self.batch_size = args.batch_size

        # Paths
        self.path = path
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        kg_file = path + '/kg_final.txt'

        # Training Args
        self.train_data, self.train_user_dict = self._load_ratings(train_file)
        self.test_data, self.test_user_dict = self._load_ratings(test_file)
        self.exist_users = self.train_user_dict.keys()
        self.n_train, self.n_test = 0, 0
        self.n_users, self.n_items = 0, 0
        self._statistic_ratings()

        # KG-related Args
        self.kg_data, self.kg_dict, self.relation_dict = self._load_kg(kg_file)
        self.n_relations, self.n_entities, self.n_triples = 0, 0, 0

        # Print out STATs of training dataset
        self.num_batch = self.n_train // self.batch_size
        self.batch_size_kg = self.n_triples // self.num_batch
        print('[n_users, n_items]=[%d, %d]' % (self.n_users, self.n_items))
        print('[n_entities, n_relations, n_triples]=[%d, %d, %d]' % (self.n_entities, self.n_relations, self.n_triples))
        print('[batch_size, batch_size_kg]=[%d, %d]' % (self.batch_size, self.batch_size_kg))
        print('[n_train, n_test]=[%d, %d]' % (self.n_train, self.n_test))


    # Load user-item interactions data
    def _load_ratings(self, file_name):
        user_dict = dict()
        inter_mat = list()

        lines = open(file_name, 'r').readlines()
        for l in lines:
            tmps = l.strip()
            inters = [int(i) for i in tmps.split(' ')]
            u_id, pos_ids = inters[0], inters[1:]
            pos_ids = list(set(pos_ids))

            for i_id in pos_ids:
                inter_mat.append([u_id, i_id])


            if len(pos_ids) > 0:
                user_dict[u_id] = pos_ids

        return np.array(inter_mat), user_dict


    #######################################################
    #######################################################

    def _statistic_ratings(self):
        self.n_users = max(max(self.train_data[:, 0]), max(self.test_data[:, 0])) + 1
        self.n_items = max(max(self.train_data[:, 1]), max(self.test_data[:, 1])) + 1
        self.n_train = len(self.train_data)
        self.n_test = len(self.test_data)


    # Reading KG info
    def _load_kg(self, file_name):

        def _construct_kg(kg_np):
            
            rd = collections.defaultdict(list)
            kg = collections.defaultdict(list)

            for head, relation, tail in kg_np:
                kg[head].append((tail, relation))
                rd[relation].append((head, tail))
            return kg, rd

        kg_np = np.loadtxt(file_name, dtype=np.int32)
        kg_np = np.unique(kg_np, axis=0)

        kg_dict, relation_dict = _construct_kg(kg_np)
        self.n_triples = len(kg_np)
        self.n_relations = max(kg_np[:, 1]) + 1
        self.n_entities = max(max(kg_np[:, 0]), max(kg_np[:, 2])) + 1

        return kg_np, kg_dict, relation_dict

    #######################################################
    #######################################################

    # Get batch data for CF aggregation
    def _generate_train_cf_batch(self):

        # fill the current batch, if users left is not enough
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        # Positive Item given a user
        def sample_pos_items_for_u(u, num):
            pos_items = self.train_user_dict[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        #######################################################
        #######################################################

        # Negative Sampling (neg_ratio=1)
        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_i_id = np.random.randint(low=0, high=self.n_items,size=1)[0]

                if neg_i_id not in self.train_user_dict[u] and neg_i_id not in neg_items:
                    neg_items.append(neg_i_id)
            return neg_items

        # Negative Sampling (neg_ratio=1)
        pos_items, neg_items = [], []
        for u in users:
            neg_items += sample_neg_items_for_u(u, 1)
            pos_items += sample_pos_items_for_u(u, 1)

        return users, pos_items, neg_items