### 数据目录
数据目录结构如下：
```bash
.
├── annotations
│   ├── labeled.json
│   ├── test_a.json
│   └── unlabeled.json
├── README.md
├── verify_data.py
└── zip_feats
    ├── labeled.zip
    ├── test_a.zip
    └── unlabeled.zip
```

请先检查数据下载是否完整，可以使用以下脚本检测 md5 
```python
python verify_data.py
```
