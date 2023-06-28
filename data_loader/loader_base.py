import os
import time
import random
import collections

import torch
import numpy as np
import pandas as pd


class DataLoaderBase(object):

    def __init__(self, args, logging):
        self.args = args
        self.data_name = args.data_name
        self.use_pretrain = args.use_pretrain
        self.pretrain_embedding_dir = args.pretrain_embedding_dir

        self.data_dir = os.path.join(args.data_dir, args.data_name)
        self.train_file = os.path.join(self.data_dir, 'train.txt')
        self.test_file = os.path.join(self.data_dir, 'test.txt')
        self.kg_file = os.path.join(self.data_dir, "kg_final.txt")

        self.cf_train_data, self.train_user_dict = self.load_cf(self.train_file)
        self.cf_test_data, self.test_user_dict = self.load_cf(self.test_file)
        self.statistic_cf()

        if self.use_pretrain == 1:
            self.load_pretrained_data()


    def load_cf(self, filename):
        user = []
        item = []
        user_dict = dict()

        lines = open(filename, 'r').readlines()
        for l in lines:
            tmp = l.strip() # Remove any trailing whitespace
            inter = [int(i) for i in tmp.split()] # Split by whitespace, Convert each part to an integer, store result in 'inter' list

            if len(inter) > 1: # if line contains more than one integer, it means a user has interacted with at least one item
                user_id, item_ids = inter[0], inter[1:] # user_id: User's ID. item_ids: Item ID which user interacted
                item_ids = list(set(item_ids)) # Convert 'item_ids' to a set, removing duplicates, then convert back to list

                for item_id in item_ids: # Iterate through each item ID that user interacted with
                    user.append(user_id) 
                    item.append(item_id)
                user_dict[user_id] = item_ids # Stores the list of unique item IDs that the user interacted with in the 'user_dict', using user's ID as the key. This keeps tracks of which items each user has intereacted with.

        user = np.array(user, dtype=np.int32)
        item = np.array(item, dtype=np.int32)
        
        return (user, item), user_dict


    def statistic_cf(self):
        self.n_users = max(max(self.cf_train_data[0]), max(self.cf_test_data[0])) + 1 # Calculate total number of users, assuming user ID starts from 0
        self.n_items = max(max(self.cf_train_data[1]), max(self.cf_test_data[1])) + 1 # Calculate total number of items
        self.n_cf_train = len(self.cf_train_data[0]) # Total number of interactions in train data
        self.n_cf_test = len(self.cf_test_data[0]) # Total number of interactions in test data


    def load_kg(self, filename):
        kg_data = pd.read_csv(filename, sep=' ', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()
        return kg_data # Each row with unique relation in the KG, with columns of h,r,t


    def sample_pos_items_for_u(self, user_dict, user_id, n_sample_pos_items):
        pos_items = user_dict[user_id] # Get list of pos_items for the given user from user_dict
        n_pos_items = len(pos_items) # Get total number of positive items

        sample_pos_items = [] # List for sampled positive items
        while True:
            if len(sample_pos_items) == n_sample_pos_items: # Break the loop once number of sampled items has reached the desired number.
                break

            pos_item_idx = np.random.randint(low=0, high=n_pos_items, size=1)[0] # Generate random index by drawing from a uniform distribution range from 0 to number of positive items.
            pos_item_id = pos_items[pos_item_idx] # Get item with the randomly generated index.
            if pos_item_id not in sample_pos_items: # Check if item has already been added to the list of sampled items. Ensure each sampled item is unique
                sample_pos_items.append(pos_item_id)
        return sample_pos_items # Return list of sampled positive items


    def sample_neg_items_for_u(self, user_dict, user_id, n_sample_neg_items):
        pos_items = user_dict[user_id] # Get list of positive items

        sample_neg_items = []
        while True:
            if len(sample_neg_items) == n_sample_neg_items:
                break

            neg_item_id = np.random.randint(low=0, high=self.n_items, size=1)[0] # Generate random item ID from uniform distribution that ranges from 0 to the total number of negative items
            if neg_item_id not in pos_items and neg_item_id not in sample_neg_items: # Check if item is a positive item or already added to the sampled items
                sample_neg_items.append(neg_item_id)
        return sample_neg_items # Return list of sampled negative items


    def generate_cf_batch(self, user_dict, batch_size): # Create a batch of data for training a collaborative filtering mode
        exist_users = user_dict.keys() # Create a list of all existing users
        if batch_size <= len(exist_users): # If batch_size if less than or euqal to the total number of users
            batch_user = random.sample(exist_users, batch_size) # Sample 'batch_size' number of users from the list of existing users
        else:
            batch_user = [random.choice(exist_users) for _ in range(batch_size)] # If 'batch_size' is larger, select 'batch_size' number of users randomly

        batch_pos_item, batch_neg_item = [], []
        for u in batch_user: # Loop through each user in the batch, and for each user:
            batch_pos_item += self.sample_pos_items_for_u(user_dict, u, 1) # Sample Positive item for the user
            batch_neg_item += self.sample_neg_items_for_u(user_dict, u, 1) # Sample Negative item for the user

        batch_user = torch.LongTensor(batch_user)
        batch_pos_item = torch.LongTensor(batch_pos_item)
        batch_neg_item = torch.LongTensor(batch_neg_item)
        return batch_user, batch_pos_item, batch_neg_item


    def sample_pos_triples_for_h(self, kg_dict, head, n_sample_pos_triples): # Sample a specified number of positive triples for a given 'head' entity
        pos_triples = kg_dict[head] # Get list of positive triples for the given head entity from the kg_dict.
        n_pos_triples = len(pos_triples) # Total number of positive triples for the head entity

        sample_relations, sample_pos_tails = [], []
        while True: 
            if len(sample_relations) == n_sample_pos_triples: 
                break

            pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0] # Draw from uniform distribution that ranges from 0 to the Number of positive triples randomly.
            tail = pos_triples[pos_triple_idx][0] # Randomly generated index for corresponding tail entity
            relation = pos_triples[pos_triple_idx][1] # Randomly generated index for corresponding Relation

            if relation not in sample_relations and tail not in sample_pos_tails: # Check if relation and tail is already added. 
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
        return sample_relations, sample_pos_tails


    def sample_neg_triples_for_h(self, kg_dict, head, relation, n_sample_neg_triples, highest_neg_idx): # Sample a specified number of Negative triples for a given 'head' entity
        pos_triples = kg_dict[head]

        sample_neg_tails = []
        while True:
            if len(sample_neg_tails) == n_sample_neg_triples:
                break

            tail = np.random.randint(low=0, high=highest_neg_idx, size=1)[0]
            if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
                sample_neg_tails.append(tail)
        return sample_neg_tails


    def generate_kg_batch(self, kg_dict, batch_size, highest_neg_idx): # Generate batch of data for training a KG
        exist_heads = kg_dict.keys() # Get keys from kg_dict, and create a list of all existing head entities
        if batch_size <= len(exist_heads): 
            batch_head = random.sample(exist_heads, batch_size) # Randomly samples batch_size number of heads from the list of existing heads
        else:
            batch_head = [random.choice(exist_heads) for _ in range(batch_size)] # Select batch_size number of heads randomly

        batch_relation, batch_pos_tail, batch_neg_tail = [], [], [] # Store relations, pos tails, and neg tails for each head in the batch
        for h in batch_head:
            relation, pos_tail = self.sample_pos_triples_for_h(kg_dict, h, 1) # Sample a positive triple for head entity
            batch_relation += relation # Append relation to the corresponding lists
            batch_pos_tail += pos_tail # Append pos tail to the corresponding lists

            neg_tail = self.sample_neg_triples_for_h(kg_dict, h, relation[0], 1, highest_neg_idx) # Sample a negative tail for head entity and the sampled relation
            batch_neg_tail += neg_tail # Append to the batch_neg_tail list

        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)
        return batch_head, batch_relation, batch_pos_tail, batch_neg_tail


    def load_pretrained_data(self):
        pre_model = 'mf'
        pretrain_path = '%s/%s/%s.npz' % (self.pretrain_embedding_dir, self.data_name, pre_model)
        pretrain_data = np.load(pretrain_path) # Load pretrained data
        self.user_pre_embed = pretrain_data['user_embed'] # Assign the User embedding
        self.item_pre_embed = pretrain_data['item_embed'] # Assign the Item embedding

        # Ensure loaded embeddings have the correct shape
        assert self.user_pre_embed.shape[0] == self.n_users
        assert self.item_pre_embed.shape[0] == self.n_items
        assert self.user_pre_embed.shape[1] == self.args.embed_dim 
        assert self.item_pre_embed.shape[1] == self.args.embed_dim