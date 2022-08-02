FROM ccr.ccs.tencentyun.com/ti_containers/pytorch:1.9.1-gpu-cu111-py38

WORKDIR /opt/ml/wxcode

# 将代码文件复制进镜像
COPY ./ ./

# 安装相应的环境（也可以放在 init.sh ）
RUN pip install -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple

# 这里无需写启动入口脚本，复现会按照 init.sh --> train.sh --> inference.sh 的顺序来执行
# CMD sh -c "sh start.sh"
