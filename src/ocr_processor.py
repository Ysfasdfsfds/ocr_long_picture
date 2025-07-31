#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR处理模块
"""

import cv2
import numpy as np
from rapidocr import RapidOCR
from typing import List, Dict, Tuple, Any
from .config import Config


class OCRProcessor:
    """OCR处理器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.engine = RapidOCR(config_path=config.rapidocr_config_path)
        self.text_score_threshold = config.text_score_threshold
    
    def process_slice(self, slice_img: np.ndarray, slice_index: int, start_y: int) -> Dict:
        """
        对单个切片进行OCR处理
        
        Args:
            slice_img: 切片图像
            slice_index: 切片索引
            start_y: 切片在原图中的起始Y坐标
            
        Returns:
            OCR处理结果
        """
        print(f"处理切片 {slice_index}...")
        
        # 进行OCR识别
        slice_img_rgb = cv2.cvtColor(slice_img, cv2.COLOR_BGR2RGB)
        result = self.engine(slice_img_rgb)
        
        # 保存可视化结果
        output_path = self.config.output_images_dir / f"slice_ocr_result_{slice_index}.jpg"
        result.vis(str(output_path))
        
        # 过滤低置信度结果
        if result.boxes is not None and result.txts is not None:
            filtered_boxes, filtered_txts, filtered_scores = self._filter_low_confidence_results(
                result.boxes, result.txts, result.scores
            )
            
            print(f"切片 {slice_index} 过滤后结果: {[(txt, score) for txt, score in zip(filtered_txts, filtered_scores)]}")
            
            if not filtered_boxes:
                print(f"切片 {slice_index} 过滤后无有效文本")
                return self._create_empty_ocr_result(slice_img.shape)
            
            # 转换坐标到原图坐标系
            adjusted_boxes = self._adjust_coordinates_to_original(filtered_boxes, start_y)
            
            # 按Y坐标排序
            sorted_results = self._sort_results_by_y(adjusted_boxes, filtered_txts, filtered_scores)
            
            return {
                'boxes': sorted_results['boxes'],
                'txts': sorted_results['txts'],
                'scores': sorted_results['scores'],
                'image_shape': slice_img.shape
            }
        else:
            print(f"切片 {slice_index} 未检测到文本")
            return self._create_empty_ocr_result(slice_img.shape)
    
    def _filter_low_confidence_results(self, boxes: List, txts: List, scores: List) -> Tuple[List, List, List]:
        """过滤低置信度结果"""
        filtered_boxes = []
        filtered_txts = []
        filtered_scores = []
        
        for box, txt, score in zip(boxes, txts, scores):
            if score >= self.text_score_threshold:
                filtered_boxes.append(box)
                filtered_txts.append(txt)
                filtered_scores.append(score)
        
        return filtered_boxes, filtered_txts, filtered_scores
    
    def _adjust_coordinates_to_original(self, boxes: List, start_y: int) -> List:
        """将坐标转换到原图坐标系"""
        adjusted_boxes = []
        for box in boxes:
            # box 格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            adjusted_box = []
            for point in box:
                adjusted_point = [point[0], point[1] + start_y]
                adjusted_box.append(adjusted_point)
            adjusted_boxes.append(adjusted_box)
        return adjusted_boxes
    
    def _sort_results_by_y(self, boxes: List, txts: List, scores: List) -> Dict:
        """按Y坐标排序结果"""
        if not boxes:
            return {'boxes': [], 'txts': [], 'scores': []}
        
        # 获取每个box的最小y坐标（即最上方的点）
        box_with_index = []
        for idx, box in enumerate(boxes):
            min_y = min(pt[1] for pt in box)
            box_with_index.append((min_y, idx, box))
        
        # 按min_y升序排序
        box_with_index.sort()
        
        # 重新排列boxes, txts, scores
        sorted_boxes = []
        sorted_txts = []
        sorted_scores = []
        for _, idx, box in box_with_index:
            sorted_boxes.append(boxes[idx])
            sorted_txts.append(txts[idx])
            sorted_scores.append(scores[idx])
        
        return {
            'boxes': sorted_boxes,
            'txts': sorted_txts,
            'scores': sorted_scores
        }
    
    def _create_empty_ocr_result(self, image_shape: Tuple) -> Dict:
        """创建空的OCR结果"""
        return {
            'boxes': [],
            'txts': [],
            'scores': [],
            'image_shape': image_shape
        }