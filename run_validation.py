#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行最终验证测试
不依赖外部服务，快速验证所有核心模块
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入并运行验证测试
from tests.test_validation import main

if __name__ == "__main__":
    sys.exit(main())
