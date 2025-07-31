#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from re import T
import requests
import json

def process_with_llm(user_question, result_to_llms):
    """
    使用本地Ollama API处理用户问题和OCR识别结果
    
    参数:
        user_question (str): 用户的问题
        result_to_llms (str): 要传递给LLM的OCR识别结果
        
    返回:
        str: LLM的回答
    """
    # 调用本地Ollama API
    ollama_url = "http://localhost:11434/api/generate"
    model_name = "qwen3:8b"
    sys_prompt = f"""
                你是一个“微信聊天内容解读助手”，你的任务是：**仅基于提供的聊天消息记录（包括群名、昵称、时间、内容等）来回答用户提出的问题**。

                你必须严格遵守以下规则：

                1. **只能基于提供的聊天记录作答，不允许主观臆测、想象或补充聊天记录中没有明确表达的信息。**
                2. **如某句话中存在指代不清（例如“这个”、“他”、“那样”），请明确标注“指代不明确”，并拒绝进行任何推测。**
                3. **若问题的答案在聊天记录中有明确依据，你必须引用相关消息，并按指定格式结构化输出。**
                4. **如未找到相关内容，请明确说明：“未在聊天记录中找到相关信息”。**
                5. **每次回答完结构化引用部分后，必须写一段简洁的总结，帮助人快速了解问题的答案。总结内容应基于引用部分，不允许添加引用之外的内容。**

                ## 聊天数据格式说明：
                聊天记录是一个字典组成的列表，**按真实聊天顺序排列**。每条记录包含一个 `type` 字段，常见取值如下：

                - `type: 'chat'`：表示一条微信消息，包含：
                - `'昵称'`：发言人昵称；若为本人，则为“我”
                - `'内容'`：发言文本内容
                - `'time'`：该条消息的时间（有些条目可能为空）

                - `type: 'time'`：表示一个时间节点（如聊天中的时间分隔线，例：“7月22日上午09:48”），是重要的上下文线索，可用于帮助理解后续聊天发生的时间。

                - `type: 'group_name'`：表示当前聊天记录所属的微信群名称，是判断聊天场景的重要依据。

                你需要综合以上三种类型的记录，准确理解聊天语境。

                ## 输出格式要求（结构化）：
                每条引用内容必须包含以下字段：
                - 说话人（“我”或他人昵称）
                - 内容
                - 时间（如记录中无对应时间节点，则标记为“未知”）
                - 上下文指向分析（如涉及引用、评价、反馈等，请说明指代谁或什么内容；如不明确则说明）

                ## 回答格式：

                1. **引用内容（结构化列表）**
                2. **总结段：简明扼要说明问题答案的核心信息，基于引用内容撰写**

                ## 示例回答：

                问题：张磊提到了哪些SDK问题？

                引用内容：
                - 说话人：张磊  
                - 内容：stop输入时调用了stop接口，此时清空缓冲区  
                - 时间：未知  
                - 上下文指向分析：这条消息说明张磊在讨论 SDK 中 stop 输入后的缓存处理机制。

                总结：张磊指出在 SDK 中，调用 stop 接口时会清空缓冲区，属于 SDK 缓存处理相关的问题。

                问题：谁说了“openai”？
                回答：未在聊天记录中找到相关信息。

                ## 聊天记录（按顺序排列）：
                {result_to_llms}

                请你根据上述聊天记录，准确回答用户提出的问题，#并必须给出结构化引用信息和总结段。
                """




    
    payload = {
        "model": model_name,
        "prompt": f"用户问题：{user_question}",
        "system": sys_prompt,
        "stream": False
    }
    
    print("正在将识别结果发送给本地Ollama的qwen3:8b模型...")
    
    try:
        response = requests.post(ollama_url, json=payload)
        response.raise_for_status()
        data = response.json()
        llm_output = data.get("response", "")
        print("Ollama模型返回结果：")
        print(llm_output)
        return llm_output
    except Exception as e:
        print("调用Ollama接口时出错：", e)
        return f"处理失败：{str(e)}"

if __name__ == "__main__":
    # 示例用法
    sample_question = "这是一个示例问题"
    sample_result = "{}"  # 这里应该是实际的JSON数据
    # 读取 output_json/modified_chat.json 并保存到变量 a
    with open("./modified_chat.json", "r", encoding="utf-8") as f:
        a = json.load(f)
    # a = a["messages"]
    while True:
        sample_question = input("请输入问题：")
        if sample_question == "exit":
            break
        
        # 确保a是消息列表，如果a是字典且有messages键，则获取messages值
        messages =  a["messages"]

        process_with_llm(sample_question, messages)