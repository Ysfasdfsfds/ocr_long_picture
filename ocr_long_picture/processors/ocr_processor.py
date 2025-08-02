#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR处理模块：专门处理OCR识别
"""

import cv2
import numpy as np
from typing import List, Dict
from rapidocr import RapidOCR

from ..config.settings import DEFAULT_TEXT_SCORE_THRESHOLD


class OCRProcessor:
    """OCR处理模块：专门处理OCR识别"""
    
    def __init__(self, config_path: str = "./config/default_rapidocr.yaml", text_score_threshold: float = DEFAULT_TEXT_SCORE_THRESHOLD):
        self.engine = RapidOCR(config_path=config_path)
        self.text_score_threshold = text_score_threshold
    
    def process_slice(self, slice_img: np.ndarray, slice_info: Dict) -> Dict:
        """处理单个切片的OCR"""
        slice_index = slice_info['slice_index']
        start_y = slice_info['start_y']
        
        print(f"处理切片 {slice_index}...")
        
        # 进行OCR识别
        slice_img_rgb = cv2.cvtColor(slice_img, cv2.COLOR_BGR2RGB)
        result = self.engine(slice_img_rgb)
        result.vis(f"output_images/slice_ocr_result_{slice_index}.jpg")
        
        # 过滤低置信度结果
        if result.boxes is not None and result.txts is not None:
            filtered_boxes, filtered_txts, filtered_scores = self._filter_low_confidence(
                result.boxes, result.txts, result.scores
            )
            
            if not filtered_boxes:
                print(f"切片 {slice_index} 过滤后无有效文本")
                return self._create_empty_result(slice_info)
            
            # 转换坐标到原图坐标系
            adjusted_boxes = self._adjust_coordinates(filtered_boxes, start_y)
            
            return {
                'slice_index': slice_index,
                'start_y': start_y,
                'end_y': slice_info['end_y'],
                'ocr_result': {
                    'boxes': adjusted_boxes,
                    'txts': filtered_txts,
                    'scores': filtered_scores,
                    'image_shape': slice_img.shape
                }
            }
        else:
            print(f"切片 {slice_index} 未检测到文本")
            return self._create_empty_result(slice_info)
    
    def _filter_low_confidence(self, boxes, txts, scores):
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
    
    def _adjust_coordinates(self, boxes, start_y):
        """转换坐标到原图坐标系"""
        adjusted_boxes = []
        for box in boxes:
            adjusted_box = []
            for point in box:
                adjusted_point = [point[0], point[1] + start_y]
                adjusted_box.append(adjusted_point)
            adjusted_boxes.append(adjusted_box)
        return adjusted_boxes
    
    def _create_empty_result(self, slice_info):
        """创建空结果"""
        return {
            'slice_index': slice_info['slice_index'],
            'start_y': slice_info['start_y'],
            'end_y': slice_info['end_y'],
            'ocr_result': {
                'boxes': [],
                'txts': [],
                'scores': [],
                'image_shape': slice_info['slice'].shape
            }
        }