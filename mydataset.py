import torch
import torch.utils.data as Data
from torch.autograd import Variable
import random
import numpy as np
import os

img_num_pf = 1000

def generate_filenames(mode):
    feat_files = []
    edge_files = []
    start_i = 0
    if mode == 'test':
        for i in range(start_i, 26000, img_num_pf):
            feat_file = 'vg_{}_{}-{}_gat_feat.txt'.format(mode, i, i+img_num_pf)
            feat_files.append(feat_file)
            edge_file = 'vg_{}_{}-{}_gat_edge_top5.txt'.format(mode, i, i+img_num_pf)
            edge_files.append(edge_file)
        feat_files.append('vg_test_26000-26446_gat_feat.txt')
        edge_files.append('vg_test_26000-26446_gat_edge_top5.txt')
    elif mode == 'train':
        for i in range(start_i, 56000, img_num_pf):
            feat_file = 'vg_{}_{}-{}_gat_feat.txt'.format(mode, i, i+img_num_pf)
            feat_files.append(feat_file)
            edge_file = 'vg_{}_{}-{}_gat_edge.txt'.format(mode, i, i+img_num_pf)
            edge_files.append(edge_file)
        feat_files.append('vg_train_56000-56224_gat_feat.txt')
        edge_files.append('vg_train_56000-56224_gat_edge.txt')
    return feat_files, edge_files


class MyDataset(Data.Dataset):
    def __init__(self, mode, feat_files, edge_files):
        """
        feat_path, edge_path: the path to the feat/edge file
        feat_files, edge_files: list of the feat/edge filenames, len(feat_files)=len(edge_files)
        file_id: current file index between (0, len(feat_files))
        feat_id, edge_id: the current line have read in feat/edge file
        feat/edge: ndarray of feat/edge
        img_id: 
        """
        self.feat_path = 'data/{}/{}_gat_feat'.format(mode, mode)
        if mode == 'train':      
            self.edge_path = 'data/{}/{}_gat_edge'.format(mode, mode)
        else:
            self.edge_path = 'data/{}/{}_gat_edge_top5'.format(mode, mode)
        self.feat_files = feat_files    # a list of filenames (in order)
        self.edge_files = edge_files
        self.file_id = 0
        self.feat_id = 0
        self.edge_id = 0
        self.feat = None
        self.edge = None
        self.img_id = 0

    def load_file(self):
        if self.file_id >= len(self.edge_files):
            return
        feat_f = open(os.path.join(self.feat_path, self.feat_files[self.file_id]), 'r')
        self.feat = feat_f.readlines()
        # edge_f = open(os.path.join(self.edge_path, self.edge_files[self.file_id]), 'r')
        # self.edge = edge_f.readlines()
        # self.feat = np.genfromtxt(os.path.join(self.feat_path, \
        #     self.feat_files[self.file_id]), dtype=np.float32)
        self.edge = np.genfromtxt(os.path.join(self.edge_path, \
            self.edge_files[self.file_id]), dtype=np.int32)
        self.feat_id = 0
        self.edge_id = 0
        self.file_id += 1

    def __len__(self):
        # return the last img idx
        last_file = self.feat_files[-1]
        return int(float(last_file.strip().split('-')[1].split('_')[0]))

    def __getitem__(self, item):
        if self.feat is None or self.feat_id >= len(self.feat):
            self.load_file()
        feat_data = []
        box_cnt = 0
        for feat_line in self.feat[self.feat_id:]:
            feat_line = feat_line.strip().split(' ')
            feat_line = np.asarray(feat_line, dtype = np.float32)
            if feat_line[1] == self.img_id:
                feat_data.append(feat_line)
                box_cnt += 1
            else:
                break
        self.feat_id += box_cnt
        self.img_id += 1

        edge_data = self.edge[self.edge_id:self.edge_id + box_cnt * 5, :]
        self.edge_id += box_cnt * 5
        
        feat_data = np.array(feat_data, dtype=np.float32)
        edge_data = np.array(edge_data, dtype=np.int32)
        return feat_data, edge_data


if __name__=="__main__":
    mode = 'train'
    feat_files, edge_files = generate_filenames(mode)
    # print(feat_files, '\n', edge_files)

    epochs = 2
    batch_size = 1

    
    for epoch in range(epochs):
        train_dataset = MyDataset(mode, feat_files, edge_files)
        train_iter = Data.DataLoader(dataset = train_dataset, batch_size = batch_size)
        for i, (x, y) in enumerate(train_iter):
            x = x[0].numpy()
            y = y[0].numpy()
            print(epoch, i, x.shape, y.shape)
            # print(x[-1,:], '\n', y[-1,:])
            # break