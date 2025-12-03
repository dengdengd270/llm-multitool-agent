Multi-Tool LLM Agent with Financial RAG and MySQL Integration

本项目实现了一个可扩展的多工具 LLM Agent，支持数学计算、天气查询、金融报告 RAG 检索，以及基于 MySQL 的 Otto 电商行为数据库查询。项目采用 LangChain 作为 Agent 框架，并兼容 DeepSeek/OpenAI 的 Chat Completions API。

一、项目功能概述

多工具 LLM Agent

自动识别用户意图

自主选择工具（calculator, weather_mock, otto_get_session, financial_report_rag）

支持 ReAct 风格规划与 Tool Calling

金融报告 RAG 系统（已集成）

使用 FAISS、BM25、bge-small-zh-v1.5 向量模型实现混合检索

自动汇总报告内容

输出带来源页码的引用

通过 legacy_rag_bridge 接入 Agent

MySQL 数据库访问工具

工具函数 otto_get_session

查询 Otto 电商行为数据（约 167 万条）

跨工具的数据流回传

可扩展的 SQL NL 工具（占位）

提供 SQL Agent 的雏形结构

未来可扩展为通用 SELECT NL2SQL Agents

CLI 运行界面

终端实时交互

支持连续对话

二、项目目录结构


```markdown
```text
.
│  .env.example
│  .gitignore
│  README.md
│  requirements.txt
│  test_rag_standalone.py
│
├─agent
│      agent_runner.py
│      llm_client.py
│      tools_math_weather.py
│      tools_otto.py
│      tools_rag_financial.py
│      __init__.py
│
├─app
│      cli_app.py
│
├─db
│      mysql_client.py
│      __init__.py
│
├─rag
│      financial_rag_core.py
│      __init__.py
│
└─storage
       bm25.pkl
       index.faiss
       meta.json

三、运行方式

安装依赖

pip install -r requirements.txt

设置环境变量

复制 .env.example 为 .env

需要配置：
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.deepseek.com/v1

OPENAI_MODEL=deepseek-chat

配置 MySQL：

DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=你的密码
DB_NAME=otto

启动 CLI 版本 Agent

python app/cli_app.py

示例：

你：1+2*3-4
Agent：3

你：今天长沙天气如何
Agent：（返回模拟天气）

你：金融危机的主要原因是什么
Agent：返回带 [报告:页码] 的金融分析

你：帮我查 Otto 数据库里 session_id 为 12899779 的数据

Agent：从 MySQL 查询并解释用户行为


