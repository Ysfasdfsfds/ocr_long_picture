#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重构后的OCR长图处理器主类
"""

from pathlib import Path
from typing import Dict

from ..processors import (
    ImageProcessor, OCRProcessor, AvatarDetector, 
    DataDeduplicator, ContentMarker
)
from ..analyzers import ChatAnalyzer
from ..exporters import ResultExporter


class RefactoredLongImageOCR:
    """重构后的长图OCR处理器主类"""
    
    def __init__(self, config_path: str = "./config/default_rapidocr.yaml"):
        self.config_path = config_path
        
        # 创建输出目录
        self.output_json_dir = Path("./output_json")
        self.output_images_dir = Path("./output_images")
        self.debug_images_dir = Path("./debug_images")
        
        self.output_json_dir.mkdir(exist_ok=True)
        self.output_images_dir.mkdir(exist_ok=True)
        self.debug_images_dir.mkdir(exist_ok=True)
        
        # 初始化各个模块
        self.image_processor = ImageProcessor()
        self.ocr_processor = OCRProcessor(config_path)
        self.avatar_detector = AvatarDetector()
        self.data_deduplicator = DataDeduplicator()
        self.result_exporter = ResultExporter(self.output_json_dir, self.output_images_dir)
        
        # 存储处理结果
        self.original_image = None
        self.all_ocr_results_original = []
        self.all_avatar_positions_original = []
        self.marked_ocr_results_original = []
        self.structured_chat_messages = []
    
    def process_long_image(self, image_path: str) -> Dict:
        """处理长图的完整流程"""
        print(f"开始处理长图: {image_path}")
        
        # 1. 切分图像
        print("步骤1: 切分图像...")
        self.original_image, slices_info = self.image_processor.slice_image(image_path)
        print(f"共切分为 {len(slices_info)} 个切片")
        
        # 2. 计算x_croped值
        print("步骤2: 计算x_croped值...")
        x_croped = self.avatar_detector.calculate_x_croped(slices_info)
        
        # 3. OCR处理和头像检测
        print("步骤3: OCR处理和头像检测...")
        all_ocr_results = []
        all_avatar_positions = []
        
        for slice_info in slices_info:
            # OCR处理
            ocr_result = self.ocr_processor.process_slice(slice_info['slice'], slice_info)
            
            # 汇总OCR结果
            if ocr_result['ocr_result']['boxes']:
                for idx, box in enumerate(ocr_result['ocr_result']['boxes']):
                    ocr_item = {
                        'slice_index': slice_info['slice_index'],
                        'box': box,
                        'text': ocr_result['ocr_result']['txts'][idx],
                        'score': ocr_result['ocr_result']['scores'][idx]
                    }
                    all_ocr_results.append(ocr_item)
            
            # 头像检测
            avatar_results = self.avatar_detector.detect_avatars_in_slice(
                slice_info['slice'], slice_info, x_croped
            )
            
            # 汇总头像结果
            for avatar_box in avatar_results:
                x, y, w, h = avatar_box
                avatar_item = {
                    'slice_index': slice_info['slice_index'],
                    'box': (x, y, w, h),
                    'center_x': x + w/2,
                    'center_y': y + h/2
                }
                all_avatar_positions.append(avatar_item)
        
        # 4. 去重处理
        print("步骤4: 去重处理...")
        deduplicated_ocr, deduplicated_avatars = self.data_deduplicator.deduplicate_results(
            all_ocr_results, all_avatar_positions
        )
        
        # 5. 内容标记
        print("步骤5: 内容标记...")
        content_marker = ContentMarker(self.original_image)
        marked_ocr_results = content_marker.mark_content_with_deduplicated_data(
            deduplicated_ocr, deduplicated_avatars
        )
        
        # 6. 聊天分析
        print("步骤6: 聊天分析...")
        marked_texts = [item.get('text', '') for item in marked_ocr_results]
        chat_analyzer = ChatAnalyzer()
        structured_messages = chat_analyzer.organize_chat_messages(marked_texts)
        
        # 7. 导出结果
        print("步骤7: 导出结果...")
        self.all_ocr_results_original = deduplicated_ocr
        self.all_avatar_positions_original = deduplicated_avatars
        self.marked_ocr_results_original = marked_ocr_results
        self.structured_chat_messages = structured_messages
        
        # 导出文件
        self.result_exporter.export_marked_ocr_results(marked_ocr_results)
        llm_input = self.result_exporter.export_structured_chat_messages(structured_messages)
        
        # 8. LLM交互 (可选)
        print("步骤8: LLM处理完成，可通过 process_with_llm() 进行交互")
        # 如需交互，可调用: process_with_llm(question, llm_input["chat_messages"])
        
        return llm_input