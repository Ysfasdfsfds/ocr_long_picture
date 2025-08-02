#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR长图处理器 - 模块化版本
项目根目录主入口文件
"""

import os
import sys
import shutil

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ocr_long_picture import RefactoredLongImageOCR
from LLM_run import process_with_llm


def main():
    """主函数"""
    print("=== OCR长图处理器 - 模块化版本 ===")
    print("项目结构已重构为模块化设计")
    
    processor = RefactoredLongImageOCR(config_path="./config/default_rapidocr.yaml")
    image_path = r"images/image_2.png"
    
    try:
        result = processor.process_long_image(image_path)
        print("\n处理结果摘要:")
        print(f"  - 聊天消息总数: {result['metadata']['total_messages']}")
        print(f"  - 普通聊天消息: {result['statistics']['nickname_messages']} 条")
        print(f"  - 时间消息: {result['statistics']['time_messages']} 条")
        print(f"  - 我的消息: {result['statistics']['my_messages']} 条")
        print(f"  - 群聊名称: {result['statistics']['group_name_messages']} 条")
        print(f"  - 撤回消息: {result['statistics']['retract_messages']} 条")
        print(f"  - 未知内容: {result['statistics']['unknown_messages']} 条")
        
        # 与LLM交互
        user_question = input("\n请输入您想询问的问题（回车跳过）：").strip()
        if user_question:
            print("\n正在与LLM交互...")
            llm_response = process_with_llm(user_question, result['messages'])
            print(f"\nLLM回答：\n{llm_response}")
        
        return result
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 清理输出目录
    if os.path.exists("output_images"):
        shutil.rmtree("output_images")
    if os.path.exists("output_json"):
        shutil.rmtree("output_json")
    
    main()