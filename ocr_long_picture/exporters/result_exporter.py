#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结果导出模块：负责结果输出
"""

import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime


class ResultExporter:
    """结果导出模块：负责结果输出"""
    
    def __init__(self, output_json_dir: Path, output_images_dir: Path):
        self.output_json_dir = output_json_dir
        self.output_images_dir = output_images_dir
        
        # 创建输出目录
        self.output_json_dir.mkdir(exist_ok=True)
        self.output_images_dir.mkdir(exist_ok=True)
    
    def export_marked_ocr_results(self, marked_ocr_results: List[Dict], output_path: str = None) -> str:
        """导出标记后的OCR结果"""
        if output_path is None:
            output_path = self.output_json_dir / "marked_ocr_results_original.json"
        
        if not marked_ocr_results:
            print("没有标记后的OCR结果可导出")
            return ""
        
        text_results = [item.get('text', '') for item in marked_ocr_results]
        
        export_data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_items": len(text_results),
                "description": "标记后的OCR文本结果 - 只包含文本内容"
            },
            "marked_texts": text_results
        }
        
        # 统计
        time_count = len([text for text in text_results if "(时间)" in text])
        nickname_count = len([text for text in text_results if "(昵称)" in text])
        content_count = len([text for text in text_results if "(内容)" in text])
        my_content_count = len([text for text in text_results if "(我的内容)" in text])
        
        export_data["statistics"] = {
            "time_items": time_count,
            "nickname_items": nickname_count,
            "content_items": content_count,
            "my_content_items": my_content_count,
            "unmarked_items": len(text_results) - time_count - nickname_count - content_count - my_content_count
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"OCR结果已导出: {output_path} (共{len(text_results)}项)")
        
        return str(output_path)
    
    def export_structured_chat_messages(self, structured_messages: List[Dict], output_path: str = None) -> Dict:
        """导出结构化聊天消息"""
        if output_path is None:
            output_path = self.output_json_dir / "structured_chat_messages.json"
        
        export_data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_messages": len(structured_messages),
                "description": "结构化聊天消息 - 按昵称、内容、时间组织"
            },
            "chat_messages": structured_messages
        }
        
        # 统计信息
        nickname_messages = len([msg for msg in structured_messages if msg.get('type') == 'chat'])
        time_messages = len([msg for msg in structured_messages if msg.get('type') == 'time'])
        my_messages = len([msg for msg in structured_messages if msg.get('type') == 'my_chat'])
        group_name_messages = len([msg for msg in structured_messages if msg.get('type') == 'group_name'])
        retract_messages = len([msg for msg in structured_messages if msg.get('type') == 'retract_message'])
        unknown_messages = len([msg for msg in structured_messages if msg.get('type') == 'unknown'])
        
        export_data["statistics"] = {
            "nickname_messages": nickname_messages,
            "time_messages": time_messages,
            "my_messages": my_messages,
            "group_name_messages": group_name_messages,
            "retract_messages": retract_messages,
            "unknown_messages": unknown_messages
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"聊天消息已导出: {output_path} (共{len(structured_messages)}条)")
        
        return export_data