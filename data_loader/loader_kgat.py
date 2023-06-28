import os
import random
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.sparse as sp

from data_loader.loader_base import DataLoaderBase


class DataLoaderKGAT(DataLoaderBase):

    def __init__(self, args, logging):
        super().__init__(args, logging)
        self.cf_batch_size = args.cf_batch_size # 1024
        self.kg_batch_size = args.kg_batch_size # 2048
        self.test_batch_size = args.test_batch_size # 10000

        kg_data = self.load_kg(self.kg_file)
        self.construct_data(kg_data)
        self.print_info(logging)

        self.laplacian_type = args.laplacian_type # Default: Random walk 
        self.create_adjacency_dict()
        self.create_adjacency_matrix()
        self.create_laplacian_dict()


    def construct_data(self, kg_data):
        # Add inverse kg data
        n_relations = max(kg_data['r']) + 1 # Number of unique relations in KG. Max value of 'r' column, and add 1
        inverse_kg_data = kg_data.copy() 
        inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns') # Swap h and t entities of each relation
        inverse_kg_data['r'] += n_relations # Relation identifiers are incremented by 'n_relations'. Ensure inverse relations have unique identifiers that are different from the original relations
        kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False) # Concat kg_data and inverse_kg_data. Reset Index. Prevent sorting

        # Re-map user id 
        kg_data['r'] += 2 # Relation identifiers are incremented by 2. 
        self.n_relations = max(kg_data['r']) + 1 # Number of max unique relations including inverse
        self.n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1 # Number of max unique entities including inverse
        self.n_users_entities = self.n_users + self.n_entities # Total number of users + entities 

        # Tuples of two arrays [User Ids][Item Ids]. 
        # Update user IDs by adding the total number of entities to each user ID.
        # Item Ids are unchanged, but converted to Integer data type
        # Output: Updated user IDs, Original item IDs
        self.cf_train_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_train_data[0]))).astype(np.int32), self.cf_train_data[1].astype(np.int32)) 
        self.cf_test_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_test_data[0]))).astype(np.int32), self.cf_test_data[1].astype(np.int32))

        # Create new dictionary where 'Each key' is the 'user ID' from the training user dictionary plus 'the total number of entities'
        # Corresponding value is the 'unique set of item IDs' associated with that user.
        # Essentially, we update user IDs to avoid overlap with entitiy IDs.
        # Also, Ensure the item IDs associated with each user are unique.
        self.train_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.train_user_dict.items()}
        self.test_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.test_user_dict.items()}

        # Add interactions to kg data
        cf2kg_train_data = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't']) # Initially populate with zeros and [h,r,t]. relation type 0. Num of rows is Num of n_cf_train
        cf2kg_train_data['h'] = self.cf_train_data[0] # Fill 'h' column with the 'user IDs' from training data
        cf2kg_train_data['t'] = self.cf_train_data[1] # Fill 't' column with the 'item IDs' from training data
    
        # Treat relations as bi-directional
        inverse_cf2kg_train_data = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't']) # Initially populated with ones and [h,r,t]. relation type 1. Num of rows is Num of n_cf_train
        inverse_cf2kg_train_data['h'] = self.cf_train_data[1] # Fill 'h' column with the 'item IDs' from training data
        inverse_cf2kg_train_data['t'] = self.cf_train_data[0] # Fill 't' column with the 'user IDs' from training data

        self.kg_train_data = pd.concat([kg_data, cf2kg_train_data, inverse_cf2kg_train_data], ignore_index=True) # Combine KG data with the user-item interaction in both directions
        self.n_kg_train = len(self.kg_train_data) # Total number of training data

        # construct kg dict
        h_list = []
        t_list = []
        r_list = []

        self.train_kg_dict = collections.defaultdict(list) # For each head entity, there is a list of pairs of tail entity and relation.
        self.train_relation_dict = collections.defaultdict(list) # For each relation, there is a list of pairs of head entity and tail entity.

        for row in self.kg_train_data.iterrows(): # Iter by row
            h, r, t = row[1] # row[0]: Key, row[1]: Data
            h_list.append(h)
            t_list.append(t)
            r_list.append(r)

            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))

        self.h_list = torch.LongTensor(h_list)
        self.t_list = torch.LongTensor(t_list)
        self.r_list = torch.LongTensor(r_list)


    def convert_coo2tensor(self, coo):          # Convert a sparse matrix in Coordinate format (COO) into a sparse tensor in PyTorch.
        values = coo.data                       # Extract non-zero values
        indices = np.vstack((coo.row, coo.col)) # Hold the indices of non-zero values

        i = torch.LongTensor(indices)   
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape)) # Create a sparse float tensor using indicies, values, and shape of original COO matrix. Torch.size(shape) converts shape into a torch


    def create_adjacency_dict(self):
        self.adjacency_dict = {}                            # Empty dictionary
        for r, ht_list in self.train_relation_dict.items(): # ht_list: list of tuples of connected head and tails
            rows = [e[0] for e in ht_list]                  # Extract head entities
            cols = [e[1] for e in ht_list]                  # Extract tail entities
            vals = [1] * len(rows)                          # Create a list of ones with length of 'rows'. Each element of 'vals' correspond to a pair of connected entities
            adj = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_users_entities, self.n_users_entities)) # Create COO sprase matrix where value is 1 if entity represented by row and col are connected by relation 'r', or 0. Matrix is Square matrix
            self.adjacency_dict[r] = adj                    # Assign created adj to the relation 'r' in dictionary adjacency_dict
        

    def create_adjacency_matrix(self):
        rows = []
        cols = []
        for r, ht_list in self.train_relation_dict.items():
            for h, t in ht_list:
                rows.append(h)
                cols.append(t)
        vals = [1] * len(rows)
        
        self.adjacency_matrix = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_users_entities, self.n_users_entities))
        

    def create_laplacian_dict(self):
        
        def random_walk_norm_lap(adj):               # Random walk norm on adj_mat, transforming the adj_mat to create a new matrix where the rows sum to 1.
            rowsum = np.array(adj.sum(axis=1))       # Calculate the sum of each row in adj_mat. Sum represents the total weight of the edges connected to each node.

            d_inv = np.power(rowsum, -1.0).flatten() # Compute the inverse degree of each node.
            d_inv[np.isinf(d_inv)] = 0               # Replace any infinite values with 0. Infinite can happen when 1/0
            d_mat_inv = sp.diags(d_inv)              # Degree matrix

            norm_adj = d_mat_inv.dot(adj)            # Multiply the degree mat by adj_mat. Divides the weight of each edge by the degree of its originating node.
            return norm_adj.tocoo()                  # Convert the norm_adj to COO sparse format. 

        if self.laplacian_type == 'random-walk':
            norm_lap_func = random_walk_norm_lap
        else:
            raise NotImplementedError

        self.laplacian_dict = {}
        for r, adj in self.adjacency_dict.items():        # Loop over the items in adj_dict. r: relation, adj: adj_mat associated with that relation
            self.laplacian_dict[r] = norm_lap_func(adj)   # Apply norm_lap_function to each adj_mat. Normalized Lap_mat associated with the relation 'r'

        A_in = sum(self.laplacian_dict.values())          # Sum up all the normlized lap mats.
        self.A_in = self.convert_coo2tensor(A_in.tocoo()) # Convert to COO Sparse mat


    def print_info(self, logging):
        logging.info('n_users:           %d' % self.n_users)
        logging.info('n_items:           %d' % self.n_items)
        logging.info('n_entities:        %d' % self.n_entities)
        logging.info('n_users_entities:  %d' % self.n_users_entities)
        logging.info('n_relations:       %d' % self.n_relations)

        logging.info('n_h_list:          %d' % len(self.h_list))
        logging.info('n_t_list:          %d' % len(self.t_list))
        logging.info('n_r_list:          %d' % len(self.r_list))

        logging.info('n_cf_train:        %d' % self.n_cf_train)
        logging.info('n_cf_test:         %d' % self.n_cf_test)

        logging.info('n_kg_train:        %d' % self.n_kg_train)