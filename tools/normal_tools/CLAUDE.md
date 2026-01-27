# CLAUDE.md - normal_tools

本目录包含项目的普通工具模块，主要用于论文搜索和下载功能。

## 当前任务：修改 arxiv_fixed.py 从 PDF 下载改为 HTML 下载

### 任务背景

目前学术网站对 PDF 下载有严格的反爬虫管控，而HTML 格式（ArXiv5）具有显著优势：
- 下载速度快（文件小 10-20 倍）
- 无严格反爬限制
- 结构化内容便于解析

### downloade任务要求

**核心变更**:
1. 下载 URL: `https://arxiv.org/pdf/{id}.pdf` → `https://arxiv.org/html/{id}`
2. 下载延迟: `random.uniform(9.0, 10.0)` → `random.uniform(0.5, 1.0)`
3. 文件扩展名: `.pdf` → `.html`
4. Content-Type 检查: `application/pdf` → `text/html`
5. 移除 PDF 特定的 Referer 和 Accept 头

**不做的修改**:
- 不添加 HTML 解析功能（仅下载原始 HTML 文件）
- 不保留 PDF 备用方案
- 不修改 Paper 类（继续使用 `Paper.extra["saved_path"]`）


### paper.py
标准化的论文数据类 `Paper`，定义了论文的核心字段结构。
