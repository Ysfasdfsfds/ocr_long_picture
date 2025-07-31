#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
长图OCR处理主模块
"""

import cv2
import numpy as np
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Any

from .config import Config
from .image_slicer import ImageSlicer
from .ocr_processor import OCRProcessor
from .avatar_detector import AvatarDetector
from .deduplicator import Deduplicator
from .content_marker import ContentMarker
from .chat_organizer import ChatOrganizer
from .data_exporter import DataExporter


class LongImageOCR:
    """长图OCR处理器"""
    
    def __init__(self, config_path: str = "default_rapidocr.yaml"):
        """
        初始化长图OCR处理器
        
        Args:
            config_path: RapidOCR配置文件路径
        """
        # 初始化配置
        self.config = Config()
        self.config.rapidocr_config_path = config_path
        
        # 初始化各个模块
        self.image_slicer = ImageSlicer(self.config)
        self.ocr_processor = OCRProcessor(self.config)
        self.avatar_detector = AvatarDetector(self.config)
        self.deduplicator = Deduplicator(self.config)
        self.content_marker = ContentMarker(self.config)
        self.chat_organizer = ChatOrganizer(self.config)
        self.data_exporter = DataExporter(self.config)
        
        # 存储处理结果
        self.original_image = None
        self.all_ocr_results_original = []
        self.all_avatar_positions_original = []
        self.marked_ocr_results_original = []
        self.structured_chat_messages = []
    
    def process_long_image(self, image_path: str) -> Dict:
        """
        处理长图的完整流程
        
        Args:
            image_path: 图像路径
            
        Returns:
            处理结果摘要
        """
        print(f"开始处理长图: {image_path}")
        
        # 1. 切分图像
        print("步骤1: 切分图像...")
        original_image, slices_info = self.image_slicer.slice_image(image_path)
        self.original_image = original_image
        self.content_marker.set_original_image(original_image)
        print(f"共切分为 {len(slices_info)} 个切片")
        
        # 2. 处理所有切片
        print("步骤2: 处理所有切片...")
        llm_input = self._process_all_slices(slices_info)
        
        # 3. 与LLM交互
        print("步骤3: 与LLM交互...")
        self._interact_with_llm(llm_input)
        
        return {
            "total_slices": len(slices_info),
            "total_ocr_items": len(self.all_ocr_results_original),
            "total_avatars": len(self.all_avatar_positions_original),
            "total_marked_items": len(self.marked_ocr_results_original),
            "total_chat_messages": len(self.structured_chat_messages)
        }
    
    def _process_all_slices(self, slices_info: List[Dict]) -> Dict:
        """处理所有切片"""
        # 收集所有OCR结果和头像位置
        all_ocr_results = []
        all_avatar_positions = []
        
        # 首先计算x_croped值
        x_croped = self._calculate_x_croped(slices_info)
        
        # 处理每个切片
        for slice_info in slices_info:
            slice_img = slice_info['slice']
            slice_index = slice_info['slice_index']
            start_y = slice_info['start_y']
            
            print(f"处理切片 {slice_index}...")
            
            # OCR处理
            ocr_result = self.ocr_processor.process_slice(slice_img, slice_index, start_y)
            
            # 如果没有OCR结果，跳过头像检测
            if not ocr_result['boxes']:
                print(f"切片 {slice_index} 无OCR结果，跳过")
                continue
            
            # 头像检测
            avatar_positions = self.avatar_detector.detect_avatars_in_slice(slice_img, slice_index, start_y)
            
            # 收集结果
            for idx, box in enumerate(ocr_result['boxes']):
                ocr_item = {
                    'slice_index': slice_index,
                    'box': box,
                    'text': ocr_result['txts'][idx],
                    'score': ocr_result['scores'][idx]
                }
                all_ocr_results.append(ocr_item)
            
            for avatar_box in avatar_positions:
                x, y, w, h = avatar_box
                avatar_item = {
                    'slice_index': slice_index,
                    'box': (x, y, w, h),
                    'center_x': x + w/2,
                    'center_y': y + h/2
                }
                all_avatar_positions.append(avatar_item)
        
        # 去重处理
        print("\n=== 开始去重处理 ===")
        deduplicated_ocr, deduplicated_avatars = self.deduplicator.deduplicate_results(
            all_ocr_results, all_avatar_positions
        )
        
        # 内容标记
        print("\n=== 基于去重后数据重新标记 ===")
        marked_ocr_results = self.content_marker.mark_content(deduplicated_ocr, deduplicated_avatars)
        
        # 保存结果到实例属性
        self.all_ocr_results_original = deduplicated_ocr
        self.all_avatar_positions_original = deduplicated_avatars
        self.marked_ocr_results_original = marked_ocr_results
        
        # 导出标记后的OCR结果
        self.data_exporter.export_marked_ocr_results(marked_ocr_results)
        
        # 整理并导出结构化的聊天消息
        llm_input = self._organize_and_export_chat_messages()
        
        # 输出统计信息
        self._print_statistics(deduplicated_ocr, deduplicated_avatars, marked_ocr_results)
        
        return llm_input
    
    def _calculate_x_croped(self, slices_info: List[Dict]) -> int:
        """计算x_croped值"""
        from .process_avatar import preprocess_and_crop_image, slice_x_croped_values
        
        # 根据切片数量决定处理逻辑
        total_slices = len(slices_info)
        if total_slices == 1:
            slices_to_process = slices_info
            print("只有一个切片，将处理所有切片")
        elif total_slices == 2:
            slices_to_process = slices_info[:1]
            print("有2个切片，将只处理第一个切片")
        else:
            slices_to_process = slices_info[1:-1]
            print(f"共有{total_slices}个切片，将处理中间{len(slices_to_process)}个切片（排除第一个和最后一个）")
        
        # 处理选定的切片
        for index, slice_info in enumerate(slices_to_process):
            img, binary, rects = preprocess_and_crop_image(slice_info['slice'], index, slice_info['start_y'])
        
        # 从slice_x_croped_values中选择合适的target_box
        all_boxes = []
        for slice_idx, target_box in slice_x_croped_values.items():
            if target_box is not None:
                if isinstance(target_box, (list, tuple)) and len(target_box) == 4:
                    x, y, w, h = target_box
                    all_boxes.append((x, y, w, h, slice_idx))
        
        print(f"总共找到 {len(all_boxes)} 个target_box")
        
        # 按x坐标排序
        all_boxes.sort(key=lambda box: box[0])
        
        x_croped = None
        if all_boxes:
            # 对最左侧前20%的box进行操作
            left_20_percent_count = max(1, int(len(all_boxes) * 0.2))
            left_boxes = all_boxes[:left_20_percent_count]
            print(f"最左侧前20%的box数量: {left_20_percent_count}")
            
            # 找到符合要求的框
            selected_box = None
            for i, (x, y, w, h, slice_idx) in enumerate(left_boxes):
                # 判断是否严格趋近于正方形（宽高比在0.8-1.2之间）
                aspect_ratio = w / h if h > 0 else 0
                is_square_like = 0.8 <= aspect_ratio <= 1.2
                
                print(f"第{i+1}个左侧框: x={x}, y={y}, w={w}, h={h}, 宽高比={aspect_ratio:.2f}, 是否趋近正方形={is_square_like}")
                
                if is_square_like:
                    selected_box = (x, y, w, h, slice_idx)
                    print(f"找到符合要求的框: 第{i+1}个左侧框，位于slice {slice_idx}")
                    break
            
            if selected_box:
                # 基于选中的框计算x_croped
                x, y, w, h, slice_idx = selected_box
                x_croped = x + w  # 使用框的右边界作为裁剪位置
                print(f"基于选中框计算的x_croped值: {x_croped}")
                
                # 绘制selected_box到原图
                if self.original_image is not None:
                    original_with_box = self.original_image.copy()
                    cv2.rectangle(original_with_box, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.imwrite(str(self.config.output_images_dir / "selected_box_original.jpg"), original_with_box)
                    print("已将selected_box绘制到原图并保存")
            else:
                print("未找到符合要求的框（最左侧前20%中没有趋近正方形的框）")
        
        return x_croped
    
    def _organize_and_export_chat_messages(self) -> Dict:
        """整理并导出结构化的聊天消息"""
        # 提取所有标记后的文本
        marked_texts = [item.get('text', '') for item in self.marked_ocr_results_original]
        
        # 整理成结构化聊天消息
        structured_messages = self.chat_organizer.organize_chat_messages(marked_texts)
        
        # 保存到类属性
        self.structured_chat_messages = structured_messages
        
        # 导出结构化聊天消息
        export_data = self.data_exporter.export_structured_chat_messages(structured_messages)
        
        return export_data
    
    def _print_statistics(self, deduplicated_ocr: List[Dict], deduplicated_avatars: List[Dict], 
                         marked_ocr_results: List[Dict]):
        """输出统计信息"""
        print(f"\n=== 去重后汇总统计 ===")
        print(f"原图坐标系统中的OCR结果总数: {len(deduplicated_ocr)}")
        print(f"原图坐标系统中的头像位置总数: {len(deduplicated_avatars)}")
        print(f"标记后的OCR结果总数: {len(marked_ocr_results)}")
        
        # 按切片显示详细信息  
        processed_slices = set(item['slice_index'] for item in deduplicated_ocr)
        for slice_idx in sorted(processed_slices):
            ocr_count = len([item for item in deduplicated_ocr if item['slice_index'] == slice_idx])
            avatar_count = len([item for item in deduplicated_avatars if item['slice_index'] == slice_idx])
            print(f"切片 {slice_idx}: OCR结果 {ocr_count} 个, 头像位置 {avatar_count} 个")
    
    def _interact_with_llm(self, llm_input: Dict):
        """与LLM交互"""
        from LLM_run import process_with_llm
        
        chat_messages = llm_input.get("chat_messages", [])
        print(f"LLM输入包含 {len(chat_messages)} 条聊天消息")
        
        while True:
            user_question = input("请输入你想问的问题（输入'退出'结束）：")
            if user_question.strip() in ["退出", "q", "Q", "exit"]:
                print("已退出与LLM的交互。")
                break
            process_with_llm(user_question, chat_messages)
    
    def get_summary_data(self) -> Dict:
        """
        获取汇总数据：原图坐标系统中的所有OCR结果、头像位置和标记后的OCR结果
        
        Returns:
            包含OCR结果、头像位置和标记后OCR结果的字典
        """
        return {
            'ocr_results_original': self.all_ocr_results_original,
            'avatar_positions_original': self.all_avatar_positions_original,
            'marked_ocr_results_original': self.marked_ocr_results_original,
            'statistics': {
                'total_ocr_items': len(self.all_ocr_results_original),
                'total_avatars': len(self.all_avatar_positions_original),
                'total_marked_ocr_items': len(self.marked_ocr_results_original),
                'processed_slices': len(set(item['slice_index'] for item in self.all_ocr_results_original)) if self.all_ocr_results_original else 0
            }
        }
    
    def export_summary_data(self, output_path: str = None) -> str:
        """
        导出汇总数据到JSON文件
        
        Args:
            output_path: 输出文件路径，如果为None则使用默认路径
            
        Returns:
            实际的输出文件路径
        """
        return self.data_exporter.export_summary_data(
            self.all_ocr_results_original,
            self.all_avatar_positions_original,
            self.marked_ocr_results_original,
            output_path
        )
