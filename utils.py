import numpy as np
import scipy.sparse as sp
import torch
import os

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(dirpath, feat_path, edge_path, feat_file, edge_file):
# def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    # print('Loading {} dataset...'.format(dataset))

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])

    # add by wby
    # idx_features_labels = np.genfromtxt('../test/test_img0.txt', dtype=np.float32)
    # print('Loading ', os.path.join(dirpath, feat_path, feat_file))
    idx_features_labels = np.genfromtxt(os.path.join(dirpath, feat_path, feat_file), dtype=np.float32)
    idx_features_labels[:, -6] /= idx_features_labels[:, -2]
    idx_features_labels[:, -5] /= idx_features_labels[:, -1]
    idx_features_labels[:, -4] /= idx_features_labels[:, -2]
    idx_features_labels[:, -3] /= idx_features_labels[:, -1]
    features = sp.csr_matrix(idx_features_labels[:, 6:-2], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, 3])   # 2:binary, 3:multi

    # # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # add by wby
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    
    # print('Loading ', os.path.join(dirpath, edge_path, edge_file))
    # edges_unordered = np.genfromtxt('../test/test_img0_edge.txt', dtype=np.int32)
    edges_unordered = np.genfromtxt(os.path.join(dirpath, edge_path, edge_file), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    # idx_train = range(140)
    # idx_val = range(200, 500)
    # idx_test = range(500, 1500)
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)
    idx = range(idx_features_labels.shape[0])
    idx = torch.LongTensor(idx)

    # return adj, features, labels, idx_train, idx_val, idx_test
    return adj, features, labels, idx


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

def evaluation_train(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    acc = correct.sum() / len(labels)
    correct = correct.type(torch.float32)
    recall_bg = sum((preds == 0).type(torch.float32) * correct) / sum((labels == 0).type(torch.float32))
    recall_nobg = sum((preds != 0).type(torch.float32) * correct) / sum((labels != 0).type(torch.float32))
    precision_bg = sum((preds == 0).type(torch.float32) * correct) / sum((preds == 0).type(torch.float32))
    precision_nobg = sum((preds != 0).type(torch.float32) * correct) / sum((preds != 0).type(torch.float32))
    return acc, recall_bg, recall_nobg, precision_bg, precision_nobg

def evaluation_test(preds, labels):
    correct = preds.eq(labels).double()
    acc = correct.sum() / len(labels)
    correct = correct.type(torch.float32)
    recall_bg = sum((preds == 0).type(torch.float32) * correct) / sum((labels == 0).type(torch.float32))
    recall_nobg = sum((preds != 0).type(torch.float32) * correct) / sum((labels != 0).type(torch.float32))
    precision_bg = sum((preds == 0).type(torch.float32) * correct) / sum((preds == 0).type(torch.float32))
    precision_nobg = sum((preds != 0).type(torch.float32) * correct) / sum((preds != 0).type(torch.float32))
    return acc, recall_bg, recall_nobg, precision_bg, precision_nobg