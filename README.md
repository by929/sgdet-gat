# sgdet-gat
代码
```
 ├── Readme.md                  
 ├── gen_samples.py              // 生成节点和边的数据
 ├── train.py                    // 训练和测试的代码
 ├── models.py                   // 模型相关文件
 ├── layers.py                   // 模型相关文件
 ├── utils.py                    // 定义相关函数，用于数据加载和测试
```

#### 生成的节点数据，每个box包含:
1. box的全局编号，即box的id——1维
2. box所在的图像编号——1维
3. box的二分类标签——1维
4. box的多分类标签，若二分类标签为0，则多分类标签为0——1维
5. box预测的label——1维
6. box预测label的相应得分——1维
7. box的'pred_boxes_fmap'——4096维
8. box的'pred_obj_score_all'——151维
9. box的'pred_boxes'(x1,y1,x2,y2)——4维
10. box所在的图像的'im_sizes'的h,w——2维  
**实际用到的特征是7,8,9, 9的特征将x1,x2除以h，y1,y2除以w进行归一化**

#### 生成的边数据:
每行由 &quot; 节点1 节点2 &quot; 表示，链接是&quot; 节点2 &rarr; 节点1 &quot;
