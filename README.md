## [2022中国高校计算机大赛-微信大数据挑战赛](https://algo.weixin.qq.com/)

### 赛题介绍
多模态短视频分类是视频理解领域的基础技术，在安全审核、推荐运营、内容搜索等领域有着非常广泛的应用。
微信视频号每天有海量的短视频创作，我们需要用算法对这些视频分类。分类体系由产品预先定义。
我们从线上抽样真实的视频号数据，并提供视频的标题、抽帧、ASR、OCR等多模态信息，以及部分人工标注，要求参赛队伍基于这些数据，训练视频分类模型。
赛题的主要挑战包括：分类的分布不均衡，无标注数据多而有标注数据少，模态缺失，层次分类等。

大赛官方网站：https://algo.weixin.qq.com/

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
# 生成的结果默认存放在 data/result.csv 位置
python inference.py
```

### 评估模型
```python
# 注意，这是线上评测代码的示例，主要目的是帮助大家理解评测逻辑
# 因为大家没有groud truth file，本地无法直接运行
python evaluate.py
```
