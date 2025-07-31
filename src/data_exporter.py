#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据导出模块
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from .config import Config


class DataExporter:
    """数据导出器"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def export_marked_ocr_results(self, marked_ocr_results: List[Dict], output_path: str = None) -> str:
        """
        导出标记后的OCR结果到JSON文件
        
        Args:
            marked_ocr_results: 标记后的OCR结果
            output_path: 输出文件路径，如果为None则使用默认路径
            
        Returns:
            实际的输出文件路径
        """
        if output_path is None:
            output_path = self.config.output_json_dir / "marked_ocr_results_original.json"
        
        if not marked_ocr_results:
            print("没有标记后的OCR结果可导出")
            return ""
        
        # 提取所有text字段
        text_results = [item.get('text', '') for item in marked_ocr_results]
        
        export_data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_items": len(text_results),
                "description": "标记后的OCR文本结果 - 只包含文本内容"
            },
            "marked_texts": text_results
        }
        
        # 按类型分类统计
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
        
        # 保存到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"标记后的OCR结果已导出到: {output_path}")
        print(f"  - 时间标记: {time_count} 项")
        print(f"  - 昵称标记: {nickname_count} 项") 
        print(f"  - 内容标记: {content_count} 项")
        print(f"  - 我的内容: {my_content_count} 项")
        print(f"  - 未标记: {export_data['statistics']['unmarked_items']} 项")
        
        return str(output_path)
    
    def export_structured_chat_messages(self, structured_messages: List[Dict], output_path: str = None) -> Dict:
        """
        导出结构化的聊天消息
        
        Args:
            structured_messages: 结构化聊天消息
            output_path: 输出文件路径，如果为None则使用默认路径
            
        Returns:
            导出数据字典
        """
        if output_path is None:
            output_path = self.config.output_json_dir / "structured_chat_messages.json"
        
        if not structured_messages:
            print("没有结构化聊天消息可导出")
            return {}
        
        # 准备导出数据
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
        
        # 保存到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"结构化聊天消息已导出到: {output_path}")
        print(f"  - 普通聊天消息: {nickname_messages} 条")
        print(f"  - 时间消息: {time_messages} 条")
        print(f"  - 我的消息: {my_messages} 条")
        print(f"  - 群聊名称: {group_name_messages} 条")
        print(f"  - 撤回消息: {retract_messages} 条")
        print(f"  - 未知内容: {unknown_messages} 条")
        
        return export_data
    
    def export_summary_data(self, ocr_results: List[Dict], avatar_positions: List[Dict], 
                           marked_ocr_results: List[Dict], output_path: str = None) -> str:
        """
        导出汇总数据到JSON文件
        
        Args:
            ocr_results: OCR结果
            avatar_positions: 头像位置
            marked_ocr_results: 标记后的OCR结果
            output_path: 输出文件路径，如果为None则使用默认路径
            
        Returns:
            实际的输出文件路径
        """
        if output_path is None:
            output_path = self.config.output_json_dir / "summary_data_original.json"
        
        summary_data = {
            'ocr_results_original': ocr_results,
            'avatar_positions_original': avatar_positions,
            'marked_ocr_results_original': marked_ocr_results,
            'statistics': {
                'total_ocr_items': len(ocr_results),
                'total_avatars': len(avatar_positions),
                'total_marked_ocr_items': len(marked_ocr_results),
                'processed_slices': len(set(item['slice_index'] for item in ocr_results)) if ocr_results else 0
            }
        }
        
        # 转换numpy类型为Python原生类型以便JSON序列化
        summary_data = self._convert_numpy_types(summary_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        print(f"汇总数据已导出到: {output_path}")
        return str(output_path)
    
    def _convert_numpy_types(self, data: Any) -> Any:
        """递归转换numpy类型为Python原生类型"""
        if isinstance(data, dict):
            return {k: self._convert_numpy_types(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_numpy_types(v) for v in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.integer):
            return int(data)
        else:
            return data