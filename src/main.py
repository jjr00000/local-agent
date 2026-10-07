import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from backend.llm_api import call_llm

def main():
    chat_history = [
        {"role":"system", "content":"你是一名Python编程助教，简洁回答代码相关问题。"}
    ]
    print("=== Bionic 本地编程助教（带上下文记忆）===")
    print("输入 exit 退出程序\n")

    while True:
        user_input = input(">>> ")
        if user_input.strip().lower() == "exit":
            print("程序退出")
            break
        
        chat_history.append({"role":"user", "content": user_input})
        reply = call_llm(chat_history, temperature=0.7, max_tokens=1024)
        print(f"\n模型回复：{reply}\n")
        chat_history.append({"role":"assistant", "content": reply})

if __name__ == "__main__":
    main()
