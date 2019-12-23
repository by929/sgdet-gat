import os
import numpy as np
import dill as pkl

import torch
import torch.utils.data as Data
from torch.autograd import Variable
import random

# 计算两个box中心点之间的距离
def dis_cnt(box1, box2):
	x1_min, y1_min, x1_max, y1_max = box1
	x2_min, y2_min, x2_max, y2_max = box2
	x1_center = (x1_min + x1_max) / 2
	y1_center = (y1_min + y1_max) / 2
	x2_center = (x2_min + x2_max) / 2
	y2_center = (y2_min + y2_max) / 2
	dis = np.sqrt(pow((x2_center - x1_center), 2) + pow((y2_center - y1_center), 2))
	return round(dis, 4)

# 计算距离矩阵
def dis_arr(boxes, box_num):
	boxes_dis = np.zeros([box_num, box_num])
	for i in range(box_num):
		boxi = boxes[i].strip().split(' ')
		boxi = np.array(boxi[-6:-2], dtype = np.float32)
		for j in range(i + 1, box_num):
			boxj = boxes[j].strip().split(' ')
			boxj = np.array(boxj[-6:-2], dtype = np.float32)
			dis_ij = dis_cnt(boxi, boxj)
			boxes_dis[i][j] = dis_ij
			boxes_dis[j][i] = dis_ij
	return boxes_dis

def generate_edge(topk, img_id):
	path = '../test'
	img_filename = 'test_img' + str(img_id) + '.txt'
	img_file = open(os.path.join(path, img_filename), 'r')
	boxes = img_file.readlines()
	boxes_start_id = int(float(boxes[0].strip().split(' ')[0]))
	# boxes = np.genfromtxt(os.path.join(path, img_filename), dtype=np.float32)
	boxes_dis = dis_arr(boxes, len(boxes))

	edge_filename = 'test_img' + str(img_id) + '_edge.txt'
	edge_file = open(os.path.join(path, edge_filename), 'w')

	boxes_dis_sort = boxes_dis.argsort(1)[:, :topk]
	for i in range(boxes_dis_sort.shape[0]):
		for j in boxes_dis_sort[i][1:]:
			edge_file.write(str(i+boxes_start_id) + ' ' + str(j+boxes_start_id) + '\n')

def generate_node(img_id):
	test_path = '../test/test_feat_relabel2/vg_test_0-1000_feat_relabel2.txt'
	test_file = open(test_path, 'r')
	test_data = test_file.readlines()

	txt_path = '../test/test_img' + str(img_id) + '.txt'
	txt_file = open(txt_path, 'w')

	idx = 0
	for line in test_data:
		data = line.strip().split(' ')
		if float(data[0]) == img_id:
			txt_file.write(str(idx) + ' ' + line)
			idx += 1
		elif float(data[0]) > img_id:
			break
		else:
			idx += 1

# # 单张图的生成
# if __name__ == '__main__':
# 	generate_node(1)	# 生成图的节点
# 	generate_edge(5, 1)	# 生成边



def generate_filenames(mode):
	filenames = []
	start_i = 0
	if mode == 'test':
		for i in range(start_i, 26000, 1000):
			# filename = 'vg_{}_{}-{}_feat_relabel2.txt'.format(mode, i, i+1000)
			filename = 'vg_{}_{}-{}_feat_relabel.txt'.format(mode, i, i+1000)
			filenames.append(filename)
		filenames.append('vg_test_26000-26446_feat_relabel2.txt')
	elif mode == 'train':
		for i in range(start_i, 56000, 1000):
			# filename = 'vg_{}_{}-{}_feat_relabel2.txt'.format(mode, i, i+1000)
			filename = 'vg_{}_{}-{}_feat_relabel.txt'.format(mode, i, i+1000)
			filenames.append(filename)
		filenames.append('vg_train_56000-56224_feat_relabel2.txt')

def generate_edge_batch(boxes, boxes_start_id, edge_file, topk):
	boxes_dis = dis_arr(boxes, len(boxes))

	boxes_dis_sort = boxes_dis.argsort(1)[:, :topk]
	for i in range(boxes_dis_sort.shape[0]):
		for j in boxes_dis_sort[i][1:]:	# 去掉节点本身
			edge_file.write(str(i+boxes_start_id) + ' ' + str(j+boxes_start_id) + '\n')

def generate_node_batch(img_nodes, feat_file):
	for line in img_nodes:
		feat_file.write(line)

# 批量生成
if __name__ == '__main__':
	topk = 5
	mode = 'train'
	# mode = 'train'
	raw_data_path = '../{}/{}_feat_relabel'.format(mode, mode)
	# raw_data_filenames = generate_filenames(mode)
	raw_data_filenames = ['vg_train_0-1000_feat_relabel.txt']
	# raw_data_filenames = ['vg_test_0-1000_feat_relabel2.txt']

	box_id = 0
	img_id = 0
	boxes_start_id = 0
	
	for filename in raw_data_filenames:
		raw_file = open(os.path.join(raw_data_path, filename), 'r')
		raw_data = raw_file.readlines()
		raw_file.close()

		img_boxes = []
		img_nodes = []
		
		new_file = 1
		for line in raw_data:
			if new_file:	# 每100张图存成1个文件
				feat_filename = 'vg_{}_{}-{}_gat_feat.txt'.format(mode, img_id, img_id + 100)
				feat_path = "./data/{}/{}_gat_feat".format(mode, mode)
				feat_file = open(os.path.join(feat_path, feat_filename), 'w')

				edge_filename = 'vg_{}_{}-{}_gat_edge.txt'.format(mode, img_id, img_id + 100)
				edge_path = "./data/{}/{}_gat_edge".format(mode, mode)
				edge_file = open(os.path.join(edge_path, edge_filename), 'w')
				new_file = 0

			data = line.strip().split(' ')
			if float(data[0]) != img_id:
				generate_edge_batch(img_boxes, boxes_start_id, edge_file, topk)
				generate_node_batch(img_nodes, feat_file)
				img_id += 1
				if img_id % 100 == 0:
					new_file = 1
				img_boxes.clear()
				img_nodes.clear()
				boxes_start_id = box_id
			img_boxes.append(line)
			img_nodes.append(str(box_id) + ' ' + line)
			box_id += 1	