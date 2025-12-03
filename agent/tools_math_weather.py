# agent/tools_math_weather.py
from langchain.tools import StructuredTool
import json

# ====== 1. Calculator 工具 ======

def calculator_core(expression: str) -> str:
    """
    简单安全版表达式计算。
    """
    try:
        if len(expression) > 100:
            return "表达式过长，无法计算。"

        allowed_chars = "0123456789+-*/(). "
        if any(ch not in allowed_chars for ch in expression):
            return "表达式包含非法字符，仅支持数字和 +-*/()."

        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算出错：{e}"


def calculator_tool_fn(expression: str) -> str:
    """
    注意：这里用的是正常的 kwargs 形式，
    LangChain 会以 calculator(expression="1+2*3") 这种方式调用。
    """
    return calculator_core(expression)


calculator_tool = StructuredTool.from_function(
    name="calculator",
    description="进行四则运算和简单数学计算，例如 '1+2*3'",
    func=calculator_tool_fn,
)


# ====== 2. Weather Mock 工具 ======

def weather_mock_core(city: str, date: str = "today") -> str:
    """
    模拟天气查询工具，真实项目可以改为调用天气 API。
    """
    city_lower = city.lower()
    if city_lower in ["tokyo", "东京"]:
        data = {
            "city": "东京",
            "date": date,
            "temp_c": 22,
            "condition": "晴转多云（模拟数据）",
        }
    else:
        data = {
            "city": city,
            "date": date,
            "temp_c": 25,
            "condition": "多云（模拟数据）",
        }
    return json.dumps(data, ensure_ascii=False)


def weather_mock_tool_fn(city: str, date: str = "today") -> str:
    """
    同样使用 kwargs 形式，LangChain 会以
    weather_mock(city="长沙", date="today") 的方式调用。
    """
    return weather_mock_core(city, date)


weather_mock_tool = StructuredTool.from_function(
    name="weather_mock",
    description="查询某城市的模拟天气信息（示例工具）",
    func=weather_mock_tool_fn,
)
