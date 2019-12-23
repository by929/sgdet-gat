from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy, evaluation_train, evaluation_test
from models import GAT, SpGAT

# Training settings
def MyParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    return args

def train(epoch):
    t = time.time()
    model.train()
    
    for i in range(0, 1000, 100):
        adj, features, labels, idx_train = load_dataset('train', i)

        if args.cuda:
            features = features.cuda()
            adj = adj.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()

        features, adj, labels = Variable(features), Variable(adj), Variable(labels)
        output = model(features, adj)

        pos_weight = torch.ones([2])
        pos_weight[1] = 10
        loss_train = F.nll_loss(output[idx_train], labels[idx_train], weight=pos_weight)

        loss_data = loss_train.data.item()
        # acc_train = accuracy(output[idx_train], labels[idx_train])
        acc_train, recall_bg, recall_nobg, precision_bg, precision_nobg \
                    = evaluation_train(output[idx_train], labels[idx_train])

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.6f}'.format(loss_train.data.item()),
              'acc_train: {:.6f}'.format(acc_train.data.item()),
              'recall_bg: {:.6f}'.format(recall_bg.data.item()),
              'recall_nobg: {:.6f}'.format(recall_nobg.data.item()),
              'precision_bg: {:.6f}'.format(precision_bg.data.item()),
              'precision_nobg: {:.6f}'.format(precision_nobg.data.item()),
              'time: {:.4f}s'.format(time.time() - t))

    return loss_data


def compute_test():
    model.eval()
    preds_test = torch.ones(1)
    labels_test = torch.ones(1)

    for i in range(0, 1000, 100):
        adj, features, labels, idx_test = load_dataset('test', 0)

        output = model(features, adj)
        pos_weight = torch.ones([2])
        pos_weight[1] = 10
        loss_test = F.nll_loss(output[idx_test], labels[idx_test], weight = pos_weight)
        acc_test = accuracy(output[idx_test], labels[idx_test])
        
        preds = output[idx_test].max(1)[1].type_as(labels)
        preds_test = preds_test.type_as(labels)
        labels_test = labels_test.type_as(labels)
        preds_test = torch.cat([preds_test, preds])
        labels_test = torch.cat([labels_test, labels[idx_test]])

        print("Test set results:",
              "loss= {:.4f}".format(loss_test.data[0]),
              "accuracy= {:.4f}".format(acc_test.data[0]))

    acc_test, recall_bg, recall_nobg, precision_bg, precision_nobg \
                    = evaluation_test(preds_test[1:], labels_test[1:])
    print('acc_test: {:.6f}'.format(acc_test.data.item()),
              'recall_bg: {:.6f}'.format(recall_bg.data.item()),
              'recall_nobg: {:.6f}'.format(recall_nobg.data.item()),
              'precision_bg: {:.6f}'.format(precision_bg.data.item()),
              'precision_nobg: {:.6f}'.format(precision_nobg.data.item()))


def load_dataset(mode, start_i):
    dirpath = './data/{}'.format(mode)
    edge_path = '{}_gat_edge'.format(mode)
    feat_path = '{}_gat_feat'.format(mode)
    feat_file = 'vg_{}_{}-{}_gat_feat.txt'.format(mode, start_i, start_i+100)
    edge_file = 'vg_{}_{}-{}_gat_edge.txt'.format(mode, start_i, start_i+100)
    adj, features, labels, idx = load_data( \
        dirpath = dirpath, feat_path = feat_path, edge_path = edge_path, \
        feat_file = feat_file, edge_file = edge_file)
    return adj, features, labels, idx


if __name__ == '__main__':
    args = MyParser()

    # Model and optimizer
    if args.sparse:
        model = SpGAT(nfeat=4251, 
                    # nfeat=features.shape[1],
                    nhid=args.hidden, 
                    # nclass=int(labels.max()) + 1, 
                    nclass=151,
                    dropout=args.dropout, 
                    nheads=args.nb_heads, 
                    alpha=args.alpha)
    else:
        model = GAT(nfeat=4251, 
                    # nfeat=features.shape[1],
                    nhid=args.hidden, 
                    # nclass=int(labels.max()) + 1, 
                    nclass=151, 
                    dropout=args.dropout, 
                    nheads=args.nb_heads, 
                    alpha=args.alpha)
    optimizer = optim.Adam(model.parameters(), 
                           lr=args.lr, 
                           weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()

    # Train model
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    
    # for epoch in range(args.epochs):
    for epoch in range(10):
        loss_values.append(train(epoch))

        torch.save(model.state_dict(), '{}.pkl'.format(epoch))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

    # Testing
    compute_test()
