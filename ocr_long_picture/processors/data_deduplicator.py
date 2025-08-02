#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据去重模块：负责重复数据处理
"""

from typing import List, Dict, Tuple

from ..config.settings import OCR_IOU_THRESHOLD, AVATAR_IOU_THRESHOLD
from ..utils.common import calculate_box_iou


class DataDeduplicator:
    """数据去重模块：负责重复数据处理"""
    
    def deduplicate_results(self, ocr_results: List[Dict], avatar_positions: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """对OCR结果和头像位置进行去重"""
        print("开始OCR结果去重...")
        deduplicated_ocr = []
        
        for current_ocr in ocr_results:
            is_duplicate = False
            current_box = current_ocr['box']
            
            for existing_ocr in deduplicated_ocr:
                existing_box = existing_ocr['box']
                iou = calculate_box_iou(current_box, existing_box)
                
                if iou > OCR_IOU_THRESHOLD:
                    is_duplicate = True
                    if current_ocr['score'] > existing_ocr['score']:
                        idx = deduplicated_ocr.index(existing_ocr)
                        deduplicated_ocr[idx] = current_ocr
                        print(f"替换重复OCR (IoU={iou:.3f}): '{existing_ocr['text']}' -> '{current_ocr['text']}'")
                    else:
                        print(f"跳过重复OCR (IoU={iou:.3f}): '{current_ocr['text']}'")
                    break
            
            if not is_duplicate:
                deduplicated_ocr.append(current_ocr)
        
        print(f"OCR去重完成: {len(ocr_results)} -> {len(deduplicated_ocr)}")
        
        print("开始头像位置去重...")
        deduplicated_avatars = []
        
        for current_avatar in avatar_positions:
            current_box = current_avatar['box']
            current_area = current_box[2] * current_box[3]
            
            duplicate_index = -1
            for j, existing_avatar in enumerate(deduplicated_avatars):
                existing_box = existing_avatar['box']
                iou = calculate_box_iou(current_box, existing_box)
                
                if iou > AVATAR_IOU_THRESHOLD:
                    duplicate_index = j
                    existing_area = existing_box[2] * existing_box[3]
                    
                    if current_area > existing_area:
                        print(f"替换重复头像 (IoU={iou:.3f}): slice_{existing_avatar['slice_index']} (面积={existing_area}) -> slice_{current_avatar['slice_index']} (面积={current_area})")
                        deduplicated_avatars[j] = current_avatar
                    else:
                        print(f"跳过重复头像 (IoU={iou:.3f}): slice_{current_avatar['slice_index']} (面积={current_area}) vs slice_{existing_avatar['slice_index']} (面积={existing_area})")
                    break
            
            if duplicate_index == -1:
                deduplicated_avatars.append(current_avatar)
        
        print(f"头像去重完成: {len(avatar_positions)} -> {len(deduplicated_avatars)}")
        
        return deduplicated_ocr, deduplicated_avatars