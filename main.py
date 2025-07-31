#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
长图OCR处理主程序
"""

import os
import shutil
from src.long_image_ocr import LongImageOCR


def main():
    """主函数"""
    # 清理输出目录
    if os.path.exists("output_images"):
        shutil.rmtree("output_images")
    if os.path.exists("output_json"):
        shutil.rmtree("output_json")
    
    # 初始化处理器
    processor = LongImageOCR(config_path="./default_rapidocr.yaml")
    
    # 处理长图
    image_path = r"images/image copy 11.png"
    
    try:
        result = processor.process_long_image(image_path)
        print("\n处理结果摘要:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        
        # 导出汇总数据
        processor.export_summary_data()
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()