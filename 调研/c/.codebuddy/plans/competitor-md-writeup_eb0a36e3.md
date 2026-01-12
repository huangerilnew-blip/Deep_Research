---
name: competitor-md-writeup
overview: Write a Competitor.md file in the workspace folder summarizing deep research tool rankings by user preference and market impact.
todos:
  - id: scan-existing-notes
    content: 使用[subagent:code-explorer]查找工作区内相关深度研究工具总结
    status: completed
  - id: draft-outline
    content: 拟定 Competitor.md 章节结构与排名维度
    status: completed
    dependencies:
      - scan-existing-notes
  - id: compose-ranking
    content: 撰写用户偏好与市场影响排名及简要说明
    status: completed
    dependencies:
      - draft-outline
  - id: add-comparison-table
    content: 整理关键指标对比表（功能、定位、定价、差异化）
    status: completed
    dependencies:
      - compose-ranking
  - id: finalize-file
    content: 校对格式与条理，保存为 Competitor.md 于工作区根目录
    status: completed
    dependencies:
      - add-comparison-table
---

## Product Overview

在当前工作区创建 Competitor.md，汇总通用深度研究工具的用户偏好与市场影响排名，并提供简明对比。

## Core Features

- 汇总通用深度研究工具名单及排名（按用户偏好与市场影响维度）
- 提供关键指标对比（功能特长、目标用户、定价模式、差异化优势）
- 输出结构化 Markdown 文档，位于工作区根目录 Competitor.md

## 方案概要

- 文件类型：Markdown 文档，包含概述、排名列表、关键指标对比、简短结论
- 组织结构：标题、排名表格/列表、差异化要点、小结，便于后续迭代更新
- 内容基准：基于现有/先前总结素材整理，保证可读性和条理性

## 计划使用的扩展

- **subagent:code-explorer**
- Purpose: 检索工作区内是否已有相关排名或素材，确保引用或补充现有内容
- Expected outcome: 确认是否存在可复用的深度研究工具排名内容，为撰写提供依据