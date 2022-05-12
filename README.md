## [2022微信大数据竞赛](https://algo.weixin.qq.com/)

### 赛题介绍
多模态短视频分类是视频理解领域的基础技术之一，在安全审核、推荐运营、内容搜索等领域有着非常广泛的应用。一条短视频中通常包含有三种模态信息，即文本、音频、视频，它们在不同语义层面的分类体系中发挥着相互促进和补充的重要作用。微信产品的内容生态繁荣，创作者覆盖范围大，导致短视频数据中普遍存在着模态缺失、相关性弱、分类标签分布不均衡等问题，是实际应用中需要着重解决的技术难点。本赛题要求参赛队伍基于微信视频号短视频数据以及对应的分类标签标注，采用合理的机器学习技术对指定的测试短视频进行分类预测。

### 数据介绍
详见 [data/README.md](data/README.md)，请确保先检查数据下载是否有缺漏错误。

### 代码介绍
- [category_id_map.py](category_id_map.py) 是category_id 和一级、二级分类的映射
- [config.py](config.py) 是配置文件
- [data_helper.py](data_helper.py) 是数据预处理模块
- [evaluate.py](evaluate.py) 是线上评测代码示例
- [inference.py](inference.py) 是生成提交文件的示例代码
- [main.py](main.py) 是训练模型的入口
- [model.py](model.py) 是baseline模型
- [util.py](util.py) 是util函数


### 安装依赖
```bash
# 安装Anaconda 和 pytorch，详情见官网：https://www.anaconda.com/ 和 https://pytorch.org/
conda install pytorch torchvision torchaudio -c pytorch

# 安装其余的依赖
pip install -r requirements.txt
```

### 训练模型
```python
python main.py
```

### 生成提交文件
```python
# 在config.py中配置ckpt_file地址后，即可运行
python inference.py
```

### 评估模型
```python
# 注意，这是线上评测代码的示例，主要目的是帮助大家理解评测逻辑
# 因为大家没有groud truth file，本地无法直接运行
python evaluate.py
```
