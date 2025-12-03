# agent/agent_runner.py
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.schema import SystemMessage
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from .tools_otto import otto_get_session_tool 
from .llm_client import get_llm
from .tools_math_weather import calculator_tool, weather_mock_tool
from .tools_rag_financial import rag_financial_tool



def create_my_agent() -> AgentExecutor:
    """
    构建一个使用多工具的 LangChain AgentExecutor。
    目前挂载的工具：
    - calculator
    - weather_mock
    - report_rag_mock（后续可替换为真实金融 RAG）
    """
    llm = get_llm()

    tools = [
    calculator_tool,
    weather_mock_tool,
    rag_financial_tool,
    otto_get_session_tool,  # 新增
    ]


    system_text = (
        "你是一个多工具 LLM Agent，可以根据用户问题自动选择合适的工具。\n"
        "- 数学计算 → 用 calculator\n"
        "- 天气相关 → 用 weather_mock\n"
        "- 金融/宏观/行业分析 → 用 financial_report_rag，从报告中检索证据并给出带[来源:页码]的回答。\n"
        "- 当用户提到 Otto、otto_test、session_id 或“这个会话里发生了什么”时，用 otto_get_session 查询该会话的原始行为数据，并据此做分析。\n"
        "工具返回结果后，请用中文做清晰的总结。"
    )


    # 用 ChatPromptTemplate 定义 prompt（新版 API 要求）
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            # 用户输入
            ("human", "{input}"),
            # Agent 中间思考 & 工具调用的 scratchpad
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # 注意：这里第三个参数是 prompt，不再是 messages=
    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,   # 调试时可以看到工具调用过程
    )
    return executor


def chat_once(agent: AgentExecutor, user_input: str) -> str:
    """
    封装一下一次对话调用，方便 CLI / Web 统一调用。
    """
    result = agent.invoke({"input": user_input})
    # LangChain 约定输出里 'output' 字段是最终回复
    return result["output"]
