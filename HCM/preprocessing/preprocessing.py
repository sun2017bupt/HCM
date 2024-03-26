import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict
import networkx as nx
import numpy as np
import sys
import pickle as pkl
import scipy.sparse as sp
import random
import os
# random.seed(72)

def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

# edges_2
def edges_hop(edges_unordered):
    record = defaultdict(list)
    edges_2 = []
    for x,y in edges_unordered:
        record[x].append(y)
    for x,y in edges_unordered:
        for p in record[y]:
            edges_2.append([x,p])
    return np.array(edges_2)


def load_data(dataset="cora"):
    print('Loading {} dataset...'.format(dataset))
    if dataset=='cora' or dataset=='citeseer' or dataset=='pubmed':
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("/root/experiment/citeseerex/numwalk10/data/otdata/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("/root/experiment/citeseerex/numwalk10/data/otdata/ind.{}.test.index".format(dataset))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))

        idx_test = test_idx_range.tolist()

        if dataset == 'cora':
            # idx_train = range(len(y) + 1068)
            # idx_val = range(len(y) + 1068, len(y) + 1068 + 500)
            idx_train = range(1299)
            idx_val = range(1299, 2165)

        elif dataset == 'citeseer':

            idx_train = range(len(y) + 1707)
            idx_val = range(len(y) + 1707, len(y) + 1707 + 500)

        elif dataset == 'pubmed':
            idx_train = range(len(y) + 18157)
            idx_val = range(len(y) + 18157, len(y) + 18157 + 500)
        
        if dataset == "citeseer":
            new_labels = []
            for lbl in labels:
                lbl = np.where(lbl == 1)[0]
                new_labels.append(lbl[0] if list(lbl) != [] else 0)
            labels = torch.LongTensor(new_labels)
        else:
            labels = torch.LongTensor(np.where(labels)[1])
    else:
        dataset_name = dataset
        graph_adjacency_list_file_path = os.path.join('/root/experiment/citeseerex/numwalk10/data', dataset_name, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join('/root/experiment/citeseerex/numwalk10/data', dataset_name,'out1_node_feature_label.txt')
        G = nx.Graph()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])
        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                                label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                                label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))
        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        features = np.array(
        [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array(
    [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
        # features = preprocess_features(features)
        if dataset_name == 'chameleon':
            idx_train = range(1700)
            idx_val = range(1701,2000)
            idx_test = range(2001,2276)
        elif dataset_name == 'cornell':
            idx_train = range(150)
            idx_val = range(151,170)
            idx_test = range(171,182)
        elif dataset_name == 'film':
            idx_train = range(5500)
            idx_val = range(5501,7000)
            idx_test = range(7001,7599)
        elif dataset_name == 'squirrel':
            idx_train = range(4500)
            idx_val = range(4501,5000)
            idx_test = range(5001,5200)
        elif dataset_name == 'wisconsin':
            idx_train = range(170)
            idx_val = range(171,220)
            idx_test = range(221,250)
        elif dataset_name == 'texas':
            idx_train = range(130)
            idx_val = range(131,160)
            idx_test = range(161,183)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features))
    labels = torch.LongTensor(labels)

    return adj, features, labels, idx_train, idx_val, idx_test

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def randomwalk_fea(adj,walk_length = 10,num_walks = 10):
    adj_matrix = adj
    # Construct Global Pseudo-Node
    adj_matrix = np.array(adj_matrix.cpu())
    a = np.ones_like(adj_matrix[0])
    preadj = np.c_[adj_matrix, a]
    a = np.ones_like(preadj[0])
    preadj = np.r_[preadj,a.reshape(1,-1)]
    
    n = preadj.shape[0]
    walks = []
        
    # for every vertex execute random walk multi times
    for node in range(n):
        walk = []
        for i in range(num_walks):
            node1 = node
            for j in range(walk_length - 1):
                neighbors = np.where(preadj[node1] > 0)[0]
                if len(neighbors) == 0:
                    break
                node1 = np.random.choice(neighbors)
                walk.append(node1)
        walks.append(walk)
    # Nan process
    for i in range(len(walks)):
        walks[i] = np.nan_to_num(walks[i],nan=0)
    walks = np.array(walks)
    for item in walks:
        item[np.isnan(item)]=0
    # one hot encoding
    identity = np.eye(252)
    for i in range(len(walks)):
        for x in walks[i]:
            identity[i][x] = 1
    identity = torch.FloatTensor(identity)
    randomadj = torch.FloatTensor(identity[:251,:251])
    return identity, randomadj


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def itertransfer(mapper):
    for k, values in mapper.items():
        for v in values:
            yield (k, v)


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import sys
import pickle as pkl
import networkx as nx
import json
from networkx.readwrite import json_graph
import pdb

def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   #adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def sys_normalized_adjacency_i(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    rowsum = (rowsum==0)*1+rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features