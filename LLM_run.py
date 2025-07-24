#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests

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
    sys_prompt = f"""你是微信聊天解读助手，需要根据提供的JSON格式微信聊天上下文回答用户的问题。JSON数据包含了不同的消息组。
    每组消息包含很多消息组，每个消息组的格式是:
        message = {{
                    'cluster_id': cluster_id,
                    'nickname': nickname,
                    'content': content,
                    'time': time_text,
                    'is_self': is_self_message,
                    'message_y': min(box['center_y'] for box in cluster_boxes),
                    'components': {{
                        'green_boxes_count': len(green_boxes),
                        'time_boxes_count': len(time_boxes),
                        'left_boxes_count': len(left_boxes)
                    }}
                }}
    如果是本人说的话，则is_self:True 并且 nickname为 本人。
    请根据上下文回答用户的问题。 上下文JSON数据：{result_to_llms}，请解读的详细一下"""
    
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
    process_with_llm(sample_question, sample_result)