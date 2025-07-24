#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试聊天消息分析功能
"""

import json
import numpy as np
from long_image_ocr import LongImageOCR

def create_mock_merged_result():
    """创建模拟的merged_result数据用于测试"""
    
    # 模拟聊天界面的文本框数据
    mock_data = {
        'boxes': [
            # 第一条消息 - 他人消息
            [[50, 100], [150, 100], [150, 120], [50, 120]],    # 昵称: "张三"
            [[350, 105], [420, 105], [420, 115], [350, 115]], # 时间: "14:30"
            [[50, 130], [300, 130], [300, 180], [50, 180]],   # 内容: "你好，今天天气不错"
            
            # 第二条消息 - 本人消息
            [[350, 220], [420, 220], [420, 230], [350, 230]], # 时间: "14:32"
            [[500, 240], [750, 240], [750, 290], [500, 290]], # 内容: "是的，很适合出门"
            
            # 第三条消息 - 他人消息
            [[50, 320], [150, 320], [150, 340], [50, 340]],   # 昵称: "李四"
            [[350, 325], [420, 325], [420, 335], [350, 335]], # 时间: "14:35"
            [[50, 350], [280, 350], [280, 400], [50, 400]],   # 内容: "我们一起去公园吧"
            
            # 误检的头像框
            [[20, 105], [40, 105], [40, 115], [20, 115]],     # 头像符号: "○"
        ],
        'txts': [
            "张三",
            "14:30",
            "你好，今天天气不错",
            "14:32", 
            "是的，很适合出门",
            "李四",
            "14:35",
            "我们一起去公园吧",
            "○"
        ],
        'scores': [0.95, 0.90, 0.88, 0.92, 0.89, 0.94, 0.91, 0.87, 0.70],
        'image_shape': (800, 800, 3)  # 假设图像尺寸
    }
    
    return mock_data

def test_chat_analysis():
    """测试聊天消息分析功能"""
    print("开始测试聊天消息分析功能...")
    
    # 创建处理器实例
    processor = LongImageOCR()
    
    # 创建模拟数据
    mock_merged_result = create_mock_merged_result()
    
    # 执行分析
    try:
        chat_result = processor.analyze_chat_messages(mock_merged_result)
        
        print("\n=== 分析结果 ===")
        print(f"总消息数: {chat_result['total_messages']}")
        
        for msg in chat_result['messages']:
            print(f"\n消息 {msg['message_id']}:")
            print(f"  昵称: {msg['nickname']}")
            print(f"  内容: {msg['content']}")
            print(f"  时间: {msg['time']}")
            print(f"  本人消息: {msg['is_self_message']}")
        
        # 保存测试结果
        with open('./output_json/test_chat_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(chat_result, f, ensure_ascii=False, indent=2)
        
        print(f"\n测试结果已保存到: ./output_json/test_chat_analysis.json")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 确保输出目录存在
    import os
    os.makedirs("./output_json", exist_ok=True)
    
    # 运行测试
    success = test_chat_analysis()
    
    if success:
        print("\n✅ 聊天消息分析功能测试通过!")
    else:
        print("\n❌ 聊天消息分析功能测试失败!") 