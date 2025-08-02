#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内容标记模块：负责内容标记和分类
"""

import cv2
import numpy as np
import re
from typing import List, Dict, Optional

from ..config.settings import (
    TIME_PATTERNS, EXCLUDE_KEYWORDS,
    GREEN_HSV_LOWER, GREEN_HSV_UPPER,
    BLUE_HSV_LOWER, BLUE_HSV_UPPER,
    WHITE_HSV_LOWER, WHITE_HSV_UPPER,
    GREEN_RATIO_THRESHOLD, BLUE_RATIO_THRESHOLD, WHITE_RATIO_THRESHOLD
)
from ..utils.common import get_box_center_y, get_box_y_min


class ContentMarker:
    """内容标记模块：负责内容标记和分类"""
    
    def __init__(self, original_image: np.ndarray = None):
        self.original_image = original_image
    
    def mark_content_with_deduplicated_data(self, deduplicated_ocr: List[Dict], deduplicated_avatars: List[Dict]) -> List[Dict]:
        """基于去重后的数据重新标记内容"""
        print("开始重新标记...")
        
        marked_results = []
        for ocr_item in deduplicated_ocr:
            marked_item = ocr_item.copy()
            marked_item['original_text'] = marked_item['text']
            marked_results.append(marked_item)
        
        sorted_avatars = sorted(deduplicated_avatars, key=lambda x: x['center_y'])
        sorted_ocr = sorted(marked_results, key=lambda x: get_box_center_y(x['box']))
        
        print(f"处理 {len(sorted_avatars)} 个头像和 {len(sorted_ocr)} 个OCR结果")
        
        # 标记时间
        self._mark_time_content(sorted_ocr)
        
        # 基于头像位置标记昵称和内容
        self._mark_nickname_and_content_with_avatars(sorted_ocr, sorted_avatars)
        
        # 重新排序
        sorted_ocr.sort(key=lambda x: get_box_center_y(x['box']))
        print(f"插入虚拟昵称后，OCR结果总数: {len(sorted_ocr)}")
        
        # 标记绿色背景的内容
        self._mark_green_content(sorted_ocr, sorted_avatars)
        
        print("重新标记完成")
        return sorted_ocr
    
    def _mark_time_content(self, ocr_results: List[Dict]):
        """标记时间内容"""
        for ocr_item in ocr_results:
            text = ocr_item['text'].strip()
            
            if len(text) > 30:
                continue
                
            if any(keyword in text for keyword in EXCLUDE_KEYWORDS):
                continue
            
            is_time = False
            for pattern in TIME_PATTERNS:
                if re.search(pattern, text):
                    match = re.search(pattern, text)
                    if match:
                        matched_length = len(match.group())
                        match_ratio = matched_length / len(text)
                        
                        if pattern.startswith('(昨天|今天|前天|明天)') or pattern.startswith('(上午|下午|早上|中午|晚上|凌晨)'):
                            if match_ratio >= 0.4:
                                is_time = True
                                break
                        else:
                            if match_ratio >= 0.6:
                                is_time = True
                                break
            
            if is_time:
                ocr_item['text'] = text + "(时间)"
                print(f"标记时间: {ocr_item['text']}")
    
    def _mark_nickname_and_content_with_avatars(self, ocr_results: List[Dict], avatars: List[Dict]):
        """基于头像位置标记昵称和内容"""
        # 绘制头像框用于调试
        if self.original_image is not None and not hasattr(self, '_avatars_drawn'):
            debug_img = self.original_image.copy()
            for idx, avatar_item in enumerate(avatars):
                avatar_box_ = avatar_item['box']
                x, y, w, h = avatar_box_
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(debug_img, f"{idx}:y={y}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            output_path = "output_images/avatars_marked_1.jpg"
            cv2.imwrite(output_path, debug_img)
            print(f"已将所有头像框绘制到原图并保存至 {output_path}")
            self._avatars_drawn = True
        
        # 收集需要插入的虚拟昵称
        virtual_nicknames_to_insert = []
        
        for i, avatar in enumerate(avatars):
            avatar_box = avatar['box']
            x_min, y_min, w, h = avatar_box
            y_max = y_min + h
            
            next_boundary = avatars[i+1]['box'][1] if i+1 < len(avatars) else float('inf')
            
            # 寻找昵称
            nickname_found = False
            
            for j, ocr_item in enumerate(ocr_results):
                if "(时间)" in ocr_item['text']:
                    continue
                    
                box_y_min = get_box_y_min(ocr_item['box'])
                
                if y_min <= box_y_min <= y_max and not nickname_found:
                    ocr_item['text'] = ocr_item['text'] + "(昵称)"
                    print(f"标记昵称: {ocr_item['text']}")
                    nickname_found = True
                    break
            
            # 如果没有找到昵称，插入虚拟昵称
            if not nickname_found:
                print(f"警告: 头像 {i} (y={y_min}-{y_max}) 附近未找到昵称，准备插入虚拟昵称")
                
                insert_index = len(ocr_results)
                for idx, ocr_item in enumerate(ocr_results):
                    item_y_min = get_box_y_min(ocr_item['box'])
                    if item_y_min > y_max:
                        insert_index = idx
                        break
                
                virtual_nickname = {
                    'text': f"未知用户{i+1}(昵称)",
                    'box': [[x_min, y_min], [x_min + w, y_min], [x_min + w, y_max], [x_min, y_max]],
                    'confidence': 0.0,
                    'slice_index': avatar.get('slice_index', -1),
                    'virtual': True,
                    'insert_index': insert_index
                }
                
                virtual_nicknames_to_insert.append(virtual_nickname)
        
        # 插入虚拟昵称
        virtual_nicknames_to_insert.sort(key=lambda x: x['insert_index'], reverse=True)
        for virtual_nickname in virtual_nicknames_to_insert:
            insert_index = virtual_nickname['insert_index']
            del virtual_nickname['insert_index']
            ocr_results.insert(insert_index, virtual_nickname)
            print(f"已插入虚拟昵称: {virtual_nickname['text']} 在位置 {insert_index}")
        
        # 标记内容
        for i, avatar in enumerate(avatars):
            avatar_box = avatar['box']
            x_min, y_min, w, h = avatar_box
            y_max = y_min + h
            
            next_boundary = avatars[i+1]['box'][1] if i+1 < len(avatars) else float('inf')
            
            # 找到对应的昵称
            nickname_index = -1
            for j, ocr_item in enumerate(ocr_results):
                if "(昵称)" in ocr_item['text']:
                    box_y_min = get_box_y_min(ocr_item['box'])
                    if y_min <= box_y_min <= y_max:
                        nickname_index = j
                        break
            
            if nickname_index >= 0:
                # 标记该昵称后的内容
                for k in range(nickname_index + 1, len(ocr_results)):
                    next_ocr = ocr_results[k]
                    if "(时间)" in next_ocr['text'] or "(昵称)" in next_ocr['text']:
                        continue
                    
                    next_box_y_min = get_box_y_min(next_ocr['box'])
                    
                    if next_box_y_min > y_min and next_box_y_min < next_boundary:
                        if "(内容)" not in next_ocr['text']:
                            next_ocr['text'] = next_ocr['text'] + "(内容)"
                            print(f"标记内容: {next_ocr['text']}")
                    elif next_box_y_min >= next_boundary:
                        break
    
    def _mark_green_content(self, ocr_results: List[Dict], avatar_positions: Optional[List[Dict]] = None):
        """标记绿色和蓝色背景的内容"""
        if self.original_image is None:
            print("原图不可用，跳过颜色内容检测")
            return
        
        my_content_boxes = []
        
        for i, ocr_item in enumerate(ocr_results):
            if "(内容)" in ocr_item['text']:
                box = ocr_item['box']
                
                is_green = False
                try:
                    is_green = self._detect_green_content_box(self.original_image, box)
                except Exception as e:
                    print(f"绿色检测失败: {e}")
                
                is_blue = False
                try:
                    is_blue = self._detect_blue_content_box(self.original_image, box)
                except Exception as e:
                    print(f"蓝色检测失败: {e}")
                
                if is_green:
                    ocr_item['text'] = ocr_item['text'].replace("(内容)", "(我的内容)")
                    my_content_boxes.append({'index': i, 'box': box})
                    print(f"标记为我的内容: {ocr_item['text']} (原因: 绿色背景)")
                elif is_blue:
                    ocr_item['text'] = ocr_item['text'].replace("(内容)", "(我的内容)")
                    my_content_boxes.append({'index': i, 'box': box})
                    print(f"标记为我的内容: {ocr_item['text']} (原因: 蓝色背景)")
    
    def _detect_green_content_box(self, image: np.ndarray, box: List) -> bool:
        """检测文本框区域是否为绿色背景"""
        try:
            points = np.array(box, dtype=np.int32)
            min_x = max(0, int(np.min(points[:, 0])))
            max_x = min(image.shape[1], int(np.max(points[:, 0])))
            min_y = max(0, int(np.min(points[:, 1])))
            max_y = min(image.shape[0], int(np.max(points[:, 1])))
            
            if max_x <= min_x or max_y <= min_y:
                return False
            
            roi = image[min_y:max_y, min_x:max_x]
            
            if roi.size == 0:
                return False
            
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            mask = cv2.inRange(hsv, np.array(GREEN_HSV_LOWER), np.array(GREEN_HSV_UPPER))
            
            green_pixels = cv2.countNonZero(mask)
            total_pixels = roi.shape[0] * roi.shape[1]
            
            if total_pixels > 0:
                green_ratio = green_pixels / total_pixels
                return green_ratio > GREEN_RATIO_THRESHOLD
            
            return False
            
        except Exception as e:
            print(f"检测绿色框时出错: {e}")
            return False
    
    def _detect_blue_content_box(self, image: np.ndarray, box: List) -> bool:
        """检测文本框区域是否为蓝色背景"""
        try:
            points = np.array(box, dtype=np.int32)
            min_x = max(0, int(np.min(points[:, 0])))
            max_x = min(image.shape[1], int(np.max(points[:, 0])))
            min_y = max(0, int(np.min(points[:, 1])))
            max_y = min(image.shape[0], int(np.max(points[:, 1])))
            
            if max_x <= min_x or max_y <= min_y:
                return False
            
            roi = image[min_y:max_y, min_x:max_x]
            
            if roi.size == 0:
                return False
            
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            blue_mask = cv2.inRange(hsv, np.array(BLUE_HSV_LOWER), np.array(BLUE_HSV_UPPER))
            white_mask = cv2.inRange(hsv, np.array(WHITE_HSV_LOWER), np.array(WHITE_HSV_UPPER))
            
            blue_pixels = cv2.countNonZero(blue_mask)
            white_pixels = cv2.countNonZero(white_mask)
            total_pixels = roi.shape[0] * roi.shape[1]
            
            if total_pixels > 0:
                blue_ratio = blue_pixels / total_pixels
                white_ratio = white_pixels / total_pixels
                
                is_blue_background = (blue_ratio > BLUE_RATIO_THRESHOLD and 
                                    white_ratio < WHITE_RATIO_THRESHOLD and 
                                    blue_ratio > white_ratio)
                
                return is_blue_background
            
            return False
            
        except Exception as e:
            print(f"检测蓝色框时出错: {e}")
            return False