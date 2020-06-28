"""Cora, citeseer, pubmed dataset.

(lingfan): following dataset loading and preprocessing code from tkipf/gcn
https://github.com/tkipf/gcn/blob/master/gcn/utils.py
"""
from __future__ import absolute_import

import numpy as np
import scipy.sparse as sp
import torch
import dgl
from NGCF.Config import config
from tqdm import tqdm
from torch.nn import init
from torch.autograd import Variable


class NaiveGraphDataset(object):
    r"""Cora citation network dataset. Nodes mean author and edges mean citation
    relationships.
    """

    def __init__(self, name, data_path_prefix='../DATA/'):
        self.name = name
        self.data_path_prefix = data_path_prefix
        self.graph = dgl.DGLGraph()
        self.n_items, self.n_users = 0, 0
        self._load()

    def _load(self):

        data_path = self.data_path_prefix + self.name + '/'

        # load data
        sub = ['train', 'test']
        for sub_name in sub:
            txt_path = data_path + '{}.txt'.format(sub_name)
            with open(txt_path) as f:
                print('load meta data from {}.txt'.format(sub_name))
                for l in tqdm(f.readlines()):
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        try:
                            items = [int(i) for i in l[1:]]
                            uid = int(l[0])
                        except Exception as e:
                            # print(e)
                            continue
                        self.n_items = max(self.n_items, max(items))
                        self.n_users = max(self.n_users, uid)

        self.n_users += 1
        self.n_items += 1
        self.user_masker = _sample_mask(range(self.n_users), self.n_users + self.n_items)
        self.item_masker = _sample_mask(range(self.n_users, self.n_users + self.n_items), self.n_users + self.n_items)
        self.graph.add_nodes(self.n_users + self.n_items)

        sub = ['train', 'test']
        for sub_name in sub:
            txt_path = data_path + '{}.txt'.format(sub_name)
            with open(txt_path) as f:
                print('load data from {}.txt'.format(sub_name))
                for l in tqdm(f.readlines()):
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        try:
                            uid = int(l[0])
                            items = [int(i) + self.n_users for i in l[1:]]
                        except Exception as e:
                            # e: 有几条记录是['uid', '']，即没有链接
                            # print(e)
                            continue
                        self.graph.add_edges(uid, items)
                        self.graph.add_edges(items, uid)

        self.train_mask = _sample_mask(range(int((self.n_users + self.n_items) * 0.6)), self.n_users + self.n_items)
        self.val_mask = _sample_mask(
            range(int((self.n_users + self.n_items) * 0.6), int((self.n_users + self.n_items) * 0.8)),
            self.n_users + self.n_items)
        self.test_mask = _sample_mask(
            range(int((self.n_users + self.n_items) * 0.8), int((self.n_users + self.n_items))),
            self.n_users + self.n_items)

        dim = config['dim_emb']
        self.features = Variable(torch.FloatTensor([[1] * dim] * (self.n_users + self.n_items)), requires_grad=True)
        init.xavier_uniform_(self.features)

        self.g = self.graph
        self.g.ndata['feat'] = self.features
        self.g.ndata['train_mask'] = self.train_mask
        self.g.ndata['val_mask'] = self.val_mask
        self.g.ndata['test_mask'] = self.test_mask
        self.g.ndata['user_mask'] = self.user_masker
        self.g.ndata['item_mask'] = self.item_masker


def _sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask


def _normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.asarray(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def _encode_onehot(labels):
    classes = list(sorted(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.asarray(list(map(classes_dict.get, labels)),
                               dtype=np.int32)
    return labels_onehot


def test():
    # d = NaiveGraphDataset(name='amazon-book', data_path_prefix='/Users/crescendo/Projects/developing/NGCF-DGL/DATA/')
    # features = Variable(torch.FloatTensor([1, 2, 3]), requires_grad=True)
    # init.xavier_uniform_(features)
    pass
