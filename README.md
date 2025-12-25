# Multi-Tool LLM Agent with Financial RAG and MySQL Integration

本项目实现了一个可扩展的多工具 LLM Agent，支持数学计算、天气查询、金融报告 RAG 检索，以及基于 MySQL 的 Otto 电商行为数据库查询与推荐结果解释。项目采用 LangChain 作为 Agent 框架，并兼容 DeepSeek/OpenAI 的 Chat Completions API，重点展示了如何将传统推荐系统结果与 LLM Agent 进行融合与解释。

## 一、项目功能概述

### 多工具 LLM Agent
- 自动识别用户意图
- 自主选择工具 (`calculator`, `weather_mock`, `otto_get_session`, `financial_report_rag`, `otto_reco_explain`)
- 支持 ReAct 风格规划与 Tool Calling
- Agent 作为统一入口，调度 RAG、数据库查询与推荐解释能力

### 金融报告 RAG 系统
- 使用 FAISS、BM25、bge-small-zh-v1.5 向量模型实现混合检索
- 面向金融、宏观、行业分析类问题
- 自动汇总报告内容
- 输出带有 `[报告:页码]` 的证据引用
- 通过 `legacy_rag_bridge` 接入 Agent

### MySQL 数据库访问工具
- **工具函数**：`otto_get_session`
- 从 MySQL 中查询 Otto 电商行为原始数据（约 167 万条）
- 支持按 `session_id` 复盘用户点击、加购、下单行为
- Agent 基于真实用户行为序列进行总结分析，为推荐解释提供真实上下文信息

### 推荐系统结果解释 (Reco Explain)
- **核心能力**：将传统推荐系统（LightGBM Ranker）输出的 Top-N 推荐结果，转化为可解释、可理解的自然语言分析。
- **数据结构**：读取离线生成的 `reco_explained.jsonl`，包含模型预测得分及核心特征（共现、最近性、热门度、会话长度等）。
- **设计原则**：不引入外部品类信息，不进行业务臆测，仅解释模型与行为层面的原因。

## 二、项目目录结构

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
│      tools_otto_reco.py
│      __init__.py
│
├─app
│      cli_app.py
│
├─data
│      reco_explained.jsonl
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
```

## 三、运行方式

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 设置环境变量

复制 `.env.example` 为 `.env`：

```ini
# LLM 配置
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=[https://api.deepseek.com/v1](https://api.deepseek.com/v1)
OPENAI_MODEL=deepseek-chat

# MySQL 配置
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=你的密码
DB_NAME=otto
```

### 3. 启动 CLI 版本 Agent

```bash
python app/cli_app.py
```

### 4. 交互示例

**用户**：`1+2*3-4`
> **Agent**：`3`

**用户**：`今天长沙天气如何`
> **Agent**：（返回模拟天气）

**用户**：`金融危机的主要原因是什么`
> **Agent**：返回带 [报告:页码] 的金融分析

**用户**：`模拟一个真实推荐场景：请你复盘 session_id=12899779 的用户行为，然后基于系统给出的 top10 推荐结果，逐条给出推荐理由，并总结整体推荐策略。`
> **Agent**：逐条解释 Top-N 推荐商品的推荐原因，并总结整体推荐策略，例如：

会话较短，推荐主要依赖商品共现关系

最近点击商品对推荐结果影响显著

热门商品用于补全候选集


