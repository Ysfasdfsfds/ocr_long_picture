#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR长图处理器 - 主入口文件
专业版本，用于处理聊天截图并进行内容分析
"""

import os
import shutil
import argparse
from pathlib import Path
from refactored_ocr_processor import RefactoredLongImageOCR


def setup_output_directories():
    """设置输出目录"""
    output_dirs = ["output_images", "output_json", "debug_images"]
    
    for dir_name in output_dirs:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.makedirs(dir_name, exist_ok=True)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="OCR长图处理器 - 聊天截图分析工具")
    parser.add_argument("--image", "-i", type=str, default="images/image_2.png", 
                        help="输入图像路径 (默认: images/image_2.png)")
    parser.add_argument("--config", "-c", type=str, default="./config/default_rapidocr.yaml",
                        help="OCR配置文件路径 (默认: ./config/default_rapidocr.yaml)")
    parser.add_argument("--no-llm",  action="store_true", 
                        help="跳过LLM交互模式")
    
    args = parser.parse_args()
    
    # 验证输入文件
    if not os.path.exists(args.image):
        print(f"错误: 找不到图像文件 {args.image}")
        return 1
    
    if not os.path.exists(args.config):
        print(f"错误: 找不到配置文件 {args.config}")
        return 1
    
    # 设置输出目录
    setup_output_directories()
    
    try:
        # 创建处理器实例
        processor = RefactoredLongImageOCR(config_path=args.config)
        
        # 如果跳过LLM模式，临时修改处理方法
        if args.no_llm:
            # 保存原始方法
            original_process = processor.process_long_image
            
            def process_without_llm(image_path):
                return processor.process_long_image(image_path)
            
            # 使用修改后的方法
            result = process_without_llm(args.image)
        else:
            # 正常处理（包含LLM交互）
            result = processor.process_long_image(args.image)
            
            # 启动LLM交互模式
            from LLM_run import process_with_llm
            print("\n" + "="*50)
            print("进入LLM交互模式 - 可以询问聊天记录相关问题")
            print("输入 '退出' 或 'q' 结束交互")
            print("="*50)
            
            while True:
                try:
                    user_question = input("\n请输入你想问的问题：")
                    if user_question.strip().lower() in ["退出", "q", "exit", "quit"]:
                        print("已退出LLM交互模式。")
                        break
                    if user_question.strip():
                        process_with_llm(user_question, result["chat_messages"])
                except KeyboardInterrupt:
                    print("\n已退出LLM交互模式。")
                    break
                except Exception as e:
                    print(f"LLM处理出错: {e}")
                    continue
        
        # 输出结果摘要
        print("\n" + "="*50)
        print("处理结果摘要:")
        print("="*50)
        print(f"  - 聊天消息总数: {result['metadata']['total_messages']}")
        print(f"  - 普通聊天消息: {result['statistics']['nickname_messages']} 条")
        print(f"  - 时间消息: {result['statistics']['time_messages']} 条")
        print(f"  - 我的消息: {result['statistics']['my_messages']} 条")
        print(f"  - 群聊名称: {result['statistics']['group_name_messages']} 条")
        print(f"  - 撤回消息: {result['statistics']['retract_messages']} 条")
        print(f"  - 未知内容: {result['statistics']['unknown_messages']} 条")
        print("="*50)
        
        return 0
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())