# -*- coding:utf-8 -*-
"""
拿别人的简单前端修改了亿点点
"""
import os.path

from flask import Flask, render_template, request, jsonify,session
from datetime import datetime
from retrieval_by_faiss import *

# 设置软连接，将图像数据文件夹链接到flask的静态文件的文件夹下，flask才能访问图片
img_root_dir = CFG.image_file_dir  # 图像数据库所在位置
static_file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'img')
if not os.path.exists(static_file_dir):
    os.makedirs(static_file_dir)

print('***创建软连接，将图像数据文件夹链接到flask的静态文件的文件夹下，flask才能访问图片***\n\n')
print(f'windows需采用管理员权限打开cmd，执行以下命令：\nmklink /D {static_file_dir} {img_root_dir}')
print(f'linux系统，执行以下命令：\nln -s {img_root_dir} {static_file_dir}')
print('***创建软连接，将图像数据文件夹链接到flask的静态文件的文件夹下，flask才能访问图片***\n\n')

# 检索模块初始化
with open(CFG.feat_mat_path, 'rb') as f:
    feat_mat = pickle.load(f)
with open(CFG.map_dict_path, 'rb') as f:
    map_dict = pickle.load(f)

ir_model = ImageRetrievalModule(CFG.index_string, CFG.feat_dim, feat_mat, map_dict,
                                CFG.clip_backbone_type, CFG.device)


app = Flask(__name__)

app.secret_key='aaa'
# 没看出来有啥用
from flask import Flask, render_template, request, jsonify, redirect, url_for
from datetime import datetime
from retrieval_by_faiss import *

# 初始化部分保持不变

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    image_file = request.files.get('image')
    text = request.form.get('text')
    if image_file:
# 保存图片并获取路径
        def save_img(file, out_dir):
            now = datetime.now()
            current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = '{}-{}'.format(current_time, file.filename)
            path_img = os.path.join(out_dir, file_name)
            file.save(path_img)
            return file_name
        path_to_img = save_img(image_file, app.static_folder)
        query_type = 'image'
        query_value = os.path.join('static',path_to_img)
    else:
        query_type = 'text'
        query_value = text

    # 执行检索
    distance_result, index_result, path_list = ir_model.retrieval_func(query_value, CFG.topk)
    if query_type=='image':
        query_value = path_to_img
    # 将结果保存到全局变量中（临时存储）
    global search_results

    resultss = []
    for distance, path in zip(distance_result, path_list):
        resultss.append({
            'path': os.path.join('img', os.path.basename(path)).replace('\\', '/'),  # 确保使用正斜杠
            'distance': float(distance)
        })

    # 将数据存储到 session 中
    session['query_type'] = query_type
    session['query_value'] = query_value
    session['results'] = resultss

    # 重定向到结果页面
    return redirect(url_for('results'))
@app.route('/results')
def results():
    query_type = session.get('query_type')
    query_value = session.get('query_value')
    result = session.get('results')

    # 渲染 results.html，并传递查询内容和结果
    return render_template('results.html', query_type=query_type, query_value=query_value, results=result)



if __name__ == '__main__':
    app.run(host='localhost', port=5000)