# -*- coding=utf-8 -*-
"""

"""
import os
config_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import torch
from easydict import EasyDict

CFG = EasyDict()

#  choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/16', 'ViT-B/32', 'ViT-L/14',  'ViT-L/14@336px')
#cn_clip available models = ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']
CFG.clip_backbone_type = 'ViT-B-16'

CFG.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# CFG.image_file_dir = r'G:\deep_learning_data\coco_2017\images\val2017'  # test
CFG.image_file_dir = r'D:\PythonProjects\image_retrieval_server\data\img'

CFG.database_dir = os.path.join(config_BASE_DIR, '..', 'data')  # 特征数据库存储路径
if not os.path.exists(CFG.database_dir):
    os.makedirs(CFG.database_dir)

postfix = '_'.join(CFG.image_file_dir.split(os.path.sep)[1:])
backbone_str = CFG.clip_backbone_type.replace('/', '_')
CFG.feat_mat_path = os.path.join(CFG.database_dir, f'feat_mat-{postfix}-{backbone_str}.pkl')
CFG.map_dict_path = os.path.join(CFG.database_dir, f'map_dict-{postfix}-{backbone_str}.pkl')

# ------------------------- index -------------------------------
CFG.index_string = 'IVF4096,PQ32x8'
# CFG.index_string = 'Flat'
CFG.feat_dim = 512 if 'B' in CFG.clip_backbone_type else 768  # CLIP输出特征维度
CFG.topk = 20