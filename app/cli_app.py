import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from agent.agent_runner import create_my_agent, chat_once


def main():
    print("=== 多工具 LLM Agent (LangChain 版本 v0) ===")
    print("目前支持：calculator / weather_mock / report_rag_mock（占位版 RAG）")
    print("输入问题，或输入 exit 退出。\n")

    agent = create_my_agent()

    while True:
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见～")
            break

        if user_input.lower() in ["exit", "quit", "q"]:
            print("再见～")
            break

        if not user_input:
            continue

        try:
            answer = chat_once(agent, user_input)
        except Exception as e:
            answer = f"调用 Agent 失败：{e}"

        print(f"\nAgent：{answer}\n")


if __name__ == "__main__":
    main()
