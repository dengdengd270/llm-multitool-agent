# agent/tools_rag_financial.py
from langchain.tools import StructuredTool
from rag.financial_rag_core import answer_question


def rag_financial_tool_fn(question: str) -> str:
    """
    给 LangChain 调用的工具函数。
    内部直接调用本地 RAG 的 answer_question。
    """
    return answer_question(question)


rag_financial_tool = StructuredTool.from_function(
    name="financial_report_rag",
    description=(
        "在央行/IMF 等金融报告知识库中检索并回答宏观/金融/风险类问题；"
        "回答中会给出[报告:页码]形式的引用。"
    ),
    func=rag_financial_tool_fn,
)
