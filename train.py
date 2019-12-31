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
import torch.utils.data as Data

from utils import load_data, accuracy, evaluation_train, evaluation_test
from models import GAT, SpGAT
from mydataset import MyDataset, generate_filenames


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


def sample_training(labels, idx_train):
    idx_train_bg = []
    idx_train_nobg = []
    for idx in idx_train:
        if labels[idx] == 0:
            idx_train_bg.append(idx)
        else:
            idx_train_nobg.append(idx)
    idx_train_bg = np.array(idx_train_bg)
    idx_train_nobg = np.array(idx_train_nobg)
    np.random.shuffle(idx_train_bg)
    idx_train_sample = np.hstack([idx_train_nobg, idx_train_bg[:len(idx_train_nobg)]])
    return idx_train_sample


def train(epoch, pos_weight = None, batch_size = 100):
    t = time.time()
    model.train()

    train_feat_files, train_edge_files = generate_filenames('train')
    train_dataset = MyDataset('train', train_feat_files, train_edge_files)
    
    train_iter = Data.DataLoader(dataset = train_dataset, batch_size = 1)
    img_num = 1000
    train_output = None
    train_labels = None

    for img_id, (x, y) in enumerate(train_iter):
        # print(img_id)
        x = x[0].numpy()
        y = y[0].numpy()
        adj, features, labels, idx_train = load_data(x, y)

        if args.cuda:
            features = features.cuda()
            adj = adj.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()

        features, adj, labels = Variable(features), Variable(adj), Variable(labels)
        output = model(features, adj)
        idx_train = sample_training(labels, idx_train)

        if train_output is None:
            train_output = output[idx_train]
            train_labels = labels[idx_train]
        else:
            train_output = torch.cat((train_output, output[idx_train]), 0)
            train_labels = torch.cat((train_labels, labels[idx_train]), 0)

        if (img_id+1) % batch_size == 0 or (img_id+1) == img_num:
            if pos_weight is None:
                loss_train = F.nll_loss(train_output, train_labels)
            else:
                loss_train = F.nll_loss(train_output, train_labels, weight=pos_weight)
            
            loss_data = loss_train.data.item()
            acc_train, recall_bg, recall_nobg, precision_bg, precision_nobg \
                        = evaluation_train(train_output, train_labels)
            
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            train_output = None
            train_labels = None

            print('Epoch: {:04d}'.format(epoch+1),
                  'loss_train: {:.6f}'.format(loss_data),
                  'acc_train: {:.6f}'.format(acc_train.data.item()),
                  'recall_bg: {:.6f}'.format(recall_bg.data.item()),
                  'recall_nobg: {:.6f}'.format(recall_nobg.data.item()),
                  'precision_bg: {:.6f}'.format(precision_bg.data.item()),
                  'precision_nobg: {:.6f}'.format(precision_nobg.data.item()),
                  'time: {:.4f}s'.format(time.time() - t))
    return loss_data

def compute_test(epoch, pos_weight = None):
    model.eval()

    test_feat_files, test_edge_files = generate_filenames('test')
    test_dataset = MyDataset('test', test_feat_files, test_edge_files)

    test_iter = Data.DataLoader(dataset = test_dataset, batch_size = 1)
    test_output = None
    test_labels = None

    for _, (x, y) in enumerate(test_iter):
        x = x[0].numpy()
        y = y[0].numpy()
        adj, features, labels, idx_test = load_data(x, y)

        if args.cuda:
            features = features.cuda()
            adj = adj.cuda()
            labels = labels.cuda()
            idx_test = idx_test.cuda()

        output = model(features, adj)
        preds = output[idx_test].max(1)[1].type_as(labels)
        if test_output is None:
            test_output = preds
            test_labels = labels[idx_test]
        else:
            test_output = torch.cat((test_output, preds), 0)
            test_labels = torch.cat((test_labels, labels[idx_test]), 0)

    acc_test, recall_bg, recall_nobg, precision_bg, precision_nobg \
                    = evaluation_test(test_output, test_labels)
    
    print('Epoch: {:04d}'.format(epoch+1),
          'acc_test: {:.6f}'.format(acc_test.data.item()),
          'recall_bg: {:.6f}'.format(recall_bg.data.item()),
          'recall_nobg: {:.6f}'.format(recall_nobg.data.item()),
          'precision_bg: {:.6f}'.format(precision_bg.data.item()),
          'precision_nobg: {:.6f}'.format(precision_nobg.data.item()))


if __name__ == '__main__':
    args = MyParser()

    # Model and optimizer
    feat_dim = 4251-4-151
    label_dim = 151

    if args.sparse:
        model = SpGAT(nfeat=feat_dim, 
                    # nfeat=features.shape[1],
                    nhid=args.hidden, 
                    # nclass=int(labels.max()) + 1, 
                    nclass=label_dim,
                    dropout=args.dropout, 
                    nheads=args.nb_heads, 
                    alpha=args.alpha)
    else:
        model = GAT(nfeat=feat_dim, 
                    # nfeat=features.shape[1],
                    nhid=args.hidden, 
                    # nclass=int(labels.max()) + 1, 
                    nclass=label_dim, 
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

    pos_weight = torch.ones([label_dim]) * 15
    pos_weight[0] = 1
    # pos_weight = None

    for epoch in range(args.epochs):
    # for epoch in range(10):
        loss_values.append(train(epoch, pos_weight))

        # Testing
        compute_test(epoch, pos_weight)

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