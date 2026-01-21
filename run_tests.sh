#!/bin/bash
# 运行集成测试脚本

echo "=========================================="
echo "Multi-Agent 深度搜索系统 - 集成测试"
echo "=========================================="

# 激活 conda 环境
echo "激活 conda 环境: agent_backend"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate agent_backend

# 检查环境
echo ""
echo "检查 Python 环境..."
which python
python --version

# 安装测试依赖
echo ""
echo "安装测试依赖..."
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-test.txt

# 运行测试
echo ""
echo "=========================================="
echo "运行集成测试..."
echo "=========================================="
pytest tests/test_integration.py -v -s

# 显示测试结果
echo ""
echo "=========================================="
echo "测试完成"
echo "=========================================="
