import numpy as np
import scipy.sparse as sp
import torch
import os


def load_data(idx_features_labels, edges_unordered):
    features = sp.csr_matrix(idx_features_labels[:, 6:4102], dtype=np.float32)

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)   # 获取节点编号
    idx_map = {j: i for i, j in enumerate(idx)}
    
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    
    # 构造邻接矩阵
    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), \
        shape=(idx_features_labels.shape[0], idx_features_labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    labels = torch.LongTensor(np.array(idx_features_labels[:, 3]).reshape(1,-1))[0]

    idx = range(idx_features_labels.shape[0])
    idx = torch.LongTensor(idx)

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