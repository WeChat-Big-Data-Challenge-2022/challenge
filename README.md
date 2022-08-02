## 代码说明

### 数据

* 仅使用大赛提供的有标注数据（10万）。
* 未使用无标注数据。

### 预训练模型

* 使用了 huggingface 上提供的 `hfl/chinese-macbert-base` 模型。链接为： https://huggingface.co/hfl/chinese-macbert-base

* 使用了 SwinTransformer 官方提供的 swin-tiny 模型。链接为：https://github.com/microsoft/Swin-Transformer

### 算法描述

* 对于视觉特征，使用 swin-tiny 提取视觉特征，并用 nextvlad 技术来聚合特征
* 对于文本特征，使用 `mac-bert` 模型来提取特征。文本仅使用标题，长度限制为64.
* 视觉和文本特征直接连接起来，通过简单的 MLP 结构去预测二级分类的 id.
* 未做模型预训练、未做多模型融合

...


### 训练流程

* 无预训练，直接在有标注数据上训练。

### 测试流程

* 划分10%的数据作为验证集。取验证集上最好的模型来做测试。
