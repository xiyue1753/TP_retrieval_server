# -*- coding:utf-8 -*-
"""
先运行并生成对应的特征缓存
"""
import torch
import cv2
import cn_clip.clip as clip
from PIL import Image
import pickle
from tqdm import tqdm

from config.base_config import CFG
from my_utils.utils import get_file_path

if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")



def main():
    clip_model, preprocess = clip.load_from_name(CFG.clip_backbone_type, device=device, download_root='D:\PythonProjects\image_retrieval_server\data\model')

    feat_list = []

    # 1. 获取图片路径
    path_img_list = get_file_path(CFG.image_file_dir, ['jpg', 'JPEG'])

    # 2. 推理
    for path_img in tqdm(path_img_list):
        image_bgr = cv2.imread(path_img)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = preprocess(Image.fromarray(image_rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            img_feat_vec = clip_model.encode_image(image)
            img_feat_vec = img_feat_vec / img_feat_vec.norm(dim=1, keepdim=True)
            img_feat_vec = img_feat_vec.cpu()

        feat_list.append(img_feat_vec)

    # 3. 存储结果
    feat_mat = torch.concat(feat_list)
    feat_mat = feat_mat.numpy()

    index_ = range(len(path_img_list))
    map_dict = dict(zip(index_, path_img_list))

    with open(CFG.feat_mat_path, 'wb') as f:
        pickle.dump(feat_mat, f)

    with open(CFG.map_dict_path, 'wb') as f:
        pickle.dump(map_dict, f)


if __name__ == '__main__':

    # filename = '000000000081.jpg'
    main()
