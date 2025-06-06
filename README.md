# Do-RAG: 模块化多模态检索增强生成系统

Do-RAG是一个强大的模块化检索增强生成（RAG）系统，支持多模态文档处理和知识图谱构建。该系统能够处理文本、表格和图像等多种类型的内容，并通过知识图谱增强检索效果。

## 核心功能

- **多模态文档处理**：支持处理文本、表格和图像等多种类型的内容
- **知识图谱构建**：自动从文档中提取实体和关系，构建知识图谱
- **模块化设计**：灵活的组件化架构，支持自定义和扩展
- **智能检索**：结合向量检索和知识图谱的混合检索策略
- **高质量生成**：基于OpenAI的LLM进行答案生成和优化

## 系统架构

系统由以下主要模块组成：

- **ingest**: 文档处理模块，负责多模态内容的解析和处理
- **kg**: 知识图谱模块，负责实体和关系的提取与存储
- **retrieval**: 检索模块，实现混合检索策略
- **generation**: 生成模块，负责答案的生成和优化
- **pipeline**: 核心流程控制模块，协调各个组件的工作

## 安装说明

1. 克隆项目仓库：
```bash
git clone [repository_url]
cd do_rag
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置OpenAI API密钥：
在`config.py`中设置你的OpenAI API密钥。

## 使用方法

基本使用示例：

```python
from app import do_rag_pipeline

# 设置文档路径和查询
file_path = "path/to/your/document.pdf"
query = "你的问题"

# 运行RAG流程
answer = do_rag_pipeline(query, file_path)
print(answer)
```

高级使用示例：

```python
from app import create_pipeline

# 创建自定义pipeline
pipeline = create_pipeline(file_path, api_key="your_api_key")

# 处理查询
query_embedding = pipeline.processor.text_model.encode([query])[0]
answer = pipeline.answer_query(query, query_embedding)
```

## 配置选项

系统支持多种配置选项：

- 内容类型过滤：可以指定只检索特定类型的内容（文本、图像、表格）
- 知识图谱构建：可选择是否启用知识图谱功能
- 检索策略：支持向量检索和知识图谱检索的混合使用

## 注意事项

- 确保有足够的OpenAI API额度
- 大文件处理可能需要较长时间
- 建议在处理大量文档时使用批处理模式

## 贡献指南

欢迎提交Issue和Pull Request来帮助改进项目。在提交代码前，请确保：

1. 代码符合项目的编码规范
2. 添加了必要的测试
3. 更新了相关文档

## 许可证

[添加许可证信息] 