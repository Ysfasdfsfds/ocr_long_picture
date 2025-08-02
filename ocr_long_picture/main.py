#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR长图处理器主入口文件
"""

import os
import shutil
from .core.main_processor import RefactoredLongImageOCR
from LLM_run import process_with_llm


def main():
    """主函数"""
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