# agent/agent_runner.py
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder

from .llm_client import get_llm
from .tools_math_weather import calculator_tool, weather_mock_tool
from .tools_rag_financial import rag_financial_tool
from .tools_otto import (
    otto_get_session_tool,
    otto_reco_explain_tool,
    otto_reco_story_tool,
)


def create_my_agent() -> AgentExecutor:
    llm = get_llm()

    tools = [
        calculator_tool,
        weather_mock_tool,
        rag_financial_tool,
        otto_get_session_tool,
        otto_reco_explain_tool,
        otto_reco_story_tool,
    ]

    system_text = (
        "你是一个多工具 LLM Agent，可以根据用户问题自动选择合适的工具。\n"
        "工具能力：\n"
        "1) 数学计算：用 calculator\n"
        "2) 天气查询：用 weather_mock\n"
        "3) 金融/宏观/行业分析：用 financial_report_rag，从报告中检索证据并给出带[来源:页码]的回答\n"
        "4) OTTO 会话行为：用 otto_get_session（按 session_id 查询 click/cart/order 原始行为并复盘）\n"
        "5) OTTO 推荐解释：\n"
        "   - 只需要解释 topN 推荐理由：用 otto_reco_explain\n"
        "   - 需要更真实的‘先复盘会话行为，再给推荐并解释理由’：优先用 otto_reco_story\n"
        "\n"
        "输出要求：\n"
        "- 只基于工具返回的结构化数据做总结，不要臆测商品属性（品类/品牌/价格）。\n"
        "- 对推荐理由，必须引用 features 或会话行为统计（例如共现强、最近性强、会话很短等），不要编造。\n"
        "- 用中文、结构清晰、分点输出。"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
    )
    return executor


def chat_once(agent: AgentExecutor, user_input: str) -> str:
    result = agent.invoke({"input": user_input})
    return result["output"]
