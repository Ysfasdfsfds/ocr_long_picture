#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
去重处理模块
"""

from typing import List, Dict, Tuple
from .config import Config


class Deduplicator:
    """去重处理器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.ocr_iou_threshold = config.ocr_iou_threshold
        self.avatar_iou_threshold = config.avatar_iou_threshold
    
    def deduplicate_results(self, ocr_results: List[Dict], avatar_positions: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        对OCR结果和头像位置进行去重
        
        Args:
            ocr_results: 原始OCR结果列表
            avatar_positions: 原始头像位置列表
            
        Returns:
            tuple: (去重后的OCR结果, 去重后的头像位置)
        """
        print("开始OCR结果去重...")
        deduplicated_ocr = self._deduplicate_ocr_results(ocr_results)
        print(f"OCR去重完成: {len(ocr_results)} -> {len(deduplicated_ocr)}")
        
        print("开始头像位置去重...")
        deduplicated_avatars = self._deduplicate_avatar_positions(avatar_positions)
        print(f"头像去重完成: {len(avatar_positions)} -> {len(deduplicated_avatars)}")
        
        return deduplicated_ocr, deduplicated_avatars
    
    def _deduplicate_ocr_results(self, ocr_results: List[Dict]) -> List[Dict]:
        """去重OCR结果"""
        deduplicated_ocr = []
        
        for i, current_ocr in enumerate(ocr_results):
            is_duplicate = False
            current_box = current_ocr['box']
            
            # 与已添加的OCR结果比较
            for existing_ocr in deduplicated_ocr:
                existing_box = existing_ocr['box']
                iou = self._calculate_box_iou(current_box, existing_box)
                
                if iou > self.ocr_iou_threshold:
                    is_duplicate = True
                    # 保留置信度更高的结果
                    if current_ocr['score'] > existing_ocr['score']:
                        # 替换现有结果
                        idx = deduplicated_ocr.index(existing_ocr)
                        deduplicated_ocr[idx] = current_ocr
                        print(f"替换重复OCR (IoU={iou:.3f}): '{existing_ocr['text']}' -> '{current_ocr['text']}'")
                    else:
                        print(f"跳过重复OCR (IoU={iou:.3f}): '{current_ocr['text']}'")
                    break
            
            if not is_duplicate:
                deduplicated_ocr.append(current_ocr)
        
        return deduplicated_ocr
    
    def _deduplicate_avatar_positions(self, avatar_positions: List[Dict]) -> List[Dict]:
        """去重头像位置"""
        deduplicated_avatars = []
        
        for i, current_avatar in enumerate(avatar_positions):
            current_box = current_avatar['box']
            current_area = current_box[2] * current_box[3]  # w * h
            
            # 检查是否与已存在的头像重复
            duplicate_index = -1
            for j, existing_avatar in enumerate(deduplicated_avatars):
                existing_box = existing_avatar['box']
                iou = self._calculate_box_iou(current_box, existing_box)
                
                if iou > self.avatar_iou_threshold:
                    duplicate_index = j
                    existing_area = existing_box[2] * existing_box[3]  # w * h
                    
                    if current_area > existing_area:
                        # 当前头像面积更大，替换已存在的
                        print(f"替换重复头像 (IoU={iou:.3f}): slice_{existing_avatar['slice_index']} (面积={existing_area}) -> slice_{current_avatar['slice_index']} (面积={current_area})")
                        deduplicated_avatars[j] = current_avatar
                    else:
                        # 已存在的头像面积更大，跳过当前头像
                        print(f"跳过重复头像 (IoU={iou:.3f}): slice_{current_avatar['slice_index']} (面积={current_area}) vs slice_{existing_avatar['slice_index']} (面积={existing_area})")
                    break
            
            # 如果没有重复，直接添加
            if duplicate_index == -1:
                deduplicated_avatars.append(current_avatar)
        
        return deduplicated_avatars
    
    def _calculate_box_iou(self, box1, box2) -> float:
        """
        计算两个矩形框的IoU (Intersection over Union)
        
        Args:
            box1, box2: 可以是两种格式
                - OCR box: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                - Avatar box: (x, y, w, h)
        
        Returns:
            float: IoU值 (0-1)
        """
        try:
            # 处理OCR box格式 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            if isinstance(box1[0], list):
                x1_min = min(pt[0] for pt in box1)
                y1_min = min(pt[1] for pt in box1)
                x1_max = max(pt[0] for pt in box1)
                y1_max = max(pt[1] for pt in box1)
            else:  # 处理Avatar box格式 (x, y, w, h)
                x1_min, y1_min, w1, h1 = box1
                x1_max = x1_min + w1
                y1_max = y1_min + h1
            
            if isinstance(box2[0], list):
                x2_min = min(pt[0] for pt in box2)
                y2_min = min(pt[1] for pt in box2)
                x2_max = max(pt[0] for pt in box2)
                y2_max = max(pt[1] for pt in box2)
            else:  # 处理Avatar box格式 (x, y, w, h)
                x2_min, y2_min, w2, h2 = box2
                x2_max = x2_min + w2
                y2_max = y2_min + h2
            
            # 计算交集
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)
            
            if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
                return 0.0
            
            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            
            # 计算并集
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)
            union_area = area1 + area2 - inter_area
            
            return inter_area / union_area if union_area > 0 else 0.0
            
        except Exception as e:
            print(f"计算IoU时出错: {e}")
            return 0.0