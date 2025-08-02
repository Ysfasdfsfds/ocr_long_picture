#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重构后的OCR长图处理器
实现模块化设计，保证输出结果与原版一致
清理版本：移除了复杂的LLM交互循环和冗余的调试输出，保留核心OCR功能
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from rapidocr import RapidOCR
import shutil
from LLM_run import process_with_llm
import re
from process_avatar import process_avatar_v2, preprocess_and_crop_image, slice_x_croped_values
from datetime import datetime


class ImageProcessor:
    """图像处理模块：负责图像切分和基础处理"""
    
    def __init__(self, slice_height: int = 1200, overlap: int = 200):
        self.slice_height = slice_height
        self.overlap = overlap
    
    def slice_image(self, image_path: str) -> Tuple[np.ndarray, List[Dict]]:
        """切分长图"""
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"无法读取图像: {image_path}")
            
        h, w, c = original_image.shape
        print(f"原始图像尺寸: {w} x {h}")
        
        if h <= self.slice_height:
            return original_image, [{
                'slice': original_image,
                'start_y': 0,
                'end_y': h,
                'slice_index': 0
            }]
        
        slices_info = []
        current_y = 0
        slice_index = 0
        
        while current_y < h:
            end_y = min(current_y + self.slice_height, h)
            slice_img = original_image[current_y:end_y, :, :]
            
            slice_info = {
                'slice': slice_img,
                'start_y': current_y,
                'end_y': end_y,
                'slice_index': slice_index
            }
            slices_info.append(slice_info)
            
            # 保存切片图像
            slice_path = Path("./output_images") / f"slice_{slice_index:03d}.jpg"
            cv2.imwrite(str(slice_path), slice_img)
            print(f"保存切片 {slice_index}: {slice_path}")
            
            if end_y >= h:
                break
            current_y = end_y - self.overlap
            slice_index += 1
            
        return original_image, slices_info


class OCRProcessor:
    """OCR处理模块：专门处理OCR识别"""
    
    def __init__(self, config_path: str = "./config/default_rapidocr.yaml", text_score_threshold: float = 0.65):
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


class AvatarDetector:
    """头像检测模块：专门处理头像识别"""
    
    def detect_avatars_in_slice(self, slice_img: np.ndarray, slice_info: Dict, x_croped: Optional[int] = None) -> List[Tuple]:
        """检测切片中的头像"""
        slice_index = slice_info['slice_index']
        start_y = slice_info['start_y']
        
        # 裁剪图像
        if x_croped is not None:
            cropped_slice = slice_img[0:slice_img.shape[0], 0:x_croped]
            print(f"切片 {slice_index} 使用x_croped={x_croped}进行裁剪")
        else:
            cropped_slice = slice_img
            print(f"切片 {slice_index} 未进行x裁剪，使用原始图像")
        
        cv2.imwrite(f"./debug_images/slice_{slice_index:03d}_avatar.jpg", cropped_slice)
        
        # 检测头像
        avatar_results = process_avatar_v2(cropped_slice)
        
        # 还原坐标到原图
        if avatar_results:
            restored_results = []
            for (x, y, w, h) in avatar_results:
                restored_box = (x, y + start_y, w, h)
                restored_results.append(restored_box)
            return restored_results
        
        return []
    
    def calculate_x_croped(self, slices_info: List[Dict]):
        """计算x_croped值"""
        # 处理切片获取target_box
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
        
        for index, slice_info in enumerate(slices_to_process):
            img, binary, rects = preprocess_and_crop_image(slice_info['slice'], index, slice_info['start_y'])
        
        # 处理slice_x_croped_values中的所有target_box
        print("开始处理slice_x_croped_values中的target_box...")
        
        # 收集所有target_box并按x坐标排序
        all_boxes = []
        for slice_idx, target_box in slice_x_croped_values.items():
            if target_box is not None:
                if isinstance(target_box, (list, tuple)) and len(target_box) == 4:
                    x, y, w, h = target_box
                    all_boxes.append((x, y, w, h, slice_idx))
        
        print(f"总共找到 {len(all_boxes)} 个target_box")
        
        # 按x坐标排序
        all_boxes.sort(key=lambda box: box[0])
        
        if not all_boxes:
            print("未找到任何target_box")
            return None
        else:
            # 找到符合要求的框
            left_20_percent_count = max(1, int(len(all_boxes) * 0.2))
            left_boxes = all_boxes[:left_20_percent_count]
            print(f"最左侧前20%的box数量: {left_20_percent_count}")
            
            selected_box = None
            for i, (x, y, w, h, slice_idx) in enumerate(left_boxes):
                aspect_ratio = w / h if h > 0 else 0
                is_square_like = 0.8 <= aspect_ratio <= 1.2
                
                print(f"第{i+1}个左侧框: x={x}, y={y}, w={w}, h={h}, 宽高比={aspect_ratio:.2f}, 是否趋近正方形={is_square_like}")
                
                if is_square_like:
                    selected_box = (x, y, w, h, slice_idx)
                    print(f"找到符合要求的框: 第{i+1}个左侧框，位于slice {slice_idx}")
                    break
            
            if selected_box:
                x, y, w, h, slice_idx = selected_box
                x_croped = x + w
                print(f"基于选中框计算的x_croped值: {x_croped}")
                return x_croped
            else:
                print("未找到符合要求的框（最左侧前20%中没有趋近正方形的框）")
                return None


class DataDeduplicator:
    """数据去重模块：负责重复数据处理"""
    
    def deduplicate_results(self, ocr_results: List[Dict], avatar_positions: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """对OCR结果和头像位置进行去重"""
        print("开始OCR结果去重...")
        deduplicated_ocr = []
        ocr_iou_threshold = 0.65
        
        for current_ocr in ocr_results:
            is_duplicate = False
            current_box = current_ocr['box']
            
            for existing_ocr in deduplicated_ocr:
                existing_box = existing_ocr['box']
                iou = self._calculate_box_iou(current_box, existing_box)
                
                if iou > ocr_iou_threshold:
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
        avatar_iou_threshold = 0.0
        
        for current_avatar in avatar_positions:
            current_box = current_avatar['box']
            current_area = current_box[2] * current_box[3]
            
            duplicate_index = -1
            for j, existing_avatar in enumerate(deduplicated_avatars):
                existing_box = existing_avatar['box']
                iou = self._calculate_box_iou(current_box, existing_box)
                
                if iou > avatar_iou_threshold:
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
    
    def _calculate_box_iou(self, box1, box2):
        """计算两个矩形框的IoU"""
        try:
            if isinstance(box1[0], list):
                x1_min = min(pt[0] for pt in box1)
                y1_min = min(pt[1] for pt in box1)
                x1_max = max(pt[0] for pt in box1)
                y1_max = max(pt[1] for pt in box1)
            else:
                x1_min, y1_min, w1, h1 = box1
                x1_max = x1_min + w1
                y1_max = y1_min + h1
            
            if isinstance(box2[0], list):
                x2_min = min(pt[0] for pt in box2)
                y2_min = min(pt[1] for pt in box2)
                x2_max = max(pt[0] for pt in box2)
                y2_max = max(pt[1] for pt in box2)
            else:
                x2_min, y2_min, w2, h2 = box2
                x2_max = x2_min + w2
                y2_max = y2_min + h2
            
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)
            
            if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
                return 0.0
            
            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)
            union_area = area1 + area2 - inter_area
            
            return inter_area / union_area if union_area > 0 else 0.0
            
        except Exception as e:
            print(f"计算IoU时出错: {e}")
            return 0.0


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
        sorted_ocr = sorted(marked_results, key=lambda x: self._get_box_center_y(x['box']))
        
        print(f"处理 {len(sorted_avatars)} 个头像和 {len(sorted_ocr)} 个OCR结果")
        
        # 标记时间
        self._mark_time_content(sorted_ocr)
        
        # 基于头像位置标记昵称和内容
        self._mark_nickname_and_content_with_avatars(sorted_ocr, sorted_avatars)
        
        # 重新排序
        sorted_ocr.sort(key=lambda x: self._get_box_center_y(x['box']))
        print(f"插入虚拟昵称后，OCR结果总数: {len(sorted_ocr)}")
        
        # 标记绿色背景的内容
        self._mark_green_content(sorted_ocr, sorted_avatars)
        
        print("重新标记完成")
        return sorted_ocr
    
    def _get_box_center_y(self, box):
        """获取box的中心Y坐标"""
        if isinstance(box[0], list):
            return sum(pt[1] for pt in box) / 4
        else:
            return box[1] + box[3] / 2
    
    def _get_box_y_min(self, box):
        """获取box的最小Y坐标"""
        if isinstance(box[0], list):
            return box[3][1]
        else:
            return box[1] + box[3] / 2
    
    def _mark_time_content(self, ocr_results: List[Dict]):
        """标记时间内容"""
        import re
        
        time_patterns = [
            r'\d{4}年\d{1,2}月\d{1,2}日\d{1,2}:\d{2}',
            r'\d{4}年\d{1,2}月\d{1,2}日',
            r'(昨天|今天|前天|明天)(早上|上午|中午|下午|晚上|凌晨)?\d{1,2}:\d{2}',
            r'(上午|下午|早上|中午|晚上|凌晨)\d{1,2}:\d{2}',
            r'\d{1,2}:\d{2}',
            r'\d{4}-\d{1,2}-\d{1,2}',
            r'\d{1,2}/\d{1,2}',
            r'\d{1,2}月\d{1,2}日',
            r'\d{1,2}:\d{2}:\d{2}',
            r'(昨天|今天|前天|明天)',
            r'周[一二三四五六日天]',
        ]
        
        for ocr_item in ocr_results:
            text = ocr_item['text'].strip()
            
            if len(text) > 30:
                continue
                
            exclude_keywords = ['报送', '回执', '会议', '参加', '人员', '工作', '通知', '安排', '要求', '地点', '内容', 
                              '完成', '需要', '前', '后', '开始', '结束', '传包', '表格', '填写', '更新', '自测']
            if any(keyword in text for keyword in exclude_keywords):
                continue
            
            is_time = False
            for pattern in time_patterns:
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
                    
                box_y_min = self._get_box_y_min(ocr_item['box'])
                
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
                    item_y_min = self._get_box_y_min(ocr_item['box'])
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
                    box_y_min = self._get_box_y_min(ocr_item['box'])
                    if y_min <= box_y_min <= y_max:
                        nickname_index = j
                        break
            
            if nickname_index >= 0:
                # 标记该昵称后的内容
                for k in range(nickname_index + 1, len(ocr_results)):
                    next_ocr = ocr_results[k]
                    if "(时间)" in next_ocr['text'] or "(昵称)" in next_ocr['text']:
                        continue
                    
                    next_box_y_min = self._get_box_y_min(next_ocr['box'])
                    
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
            
            lower_green1 = np.array([35, 40, 40])
            upper_green1 = np.array([85, 255, 255])
            
            mask = cv2.inRange(hsv, lower_green1, upper_green1)
            
            green_pixels = cv2.countNonZero(mask)
            total_pixels = roi.shape[0] * roi.shape[1]
            
            if total_pixels > 0:
                green_ratio = green_pixels / total_pixels
                return green_ratio > 0.2
            
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
            
            lower_blue = np.array([100, 30, 80])
            upper_blue = np.array([130, 180, 255])
            
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            
            blue_pixels = cv2.countNonZero(blue_mask)
            white_pixels = cv2.countNonZero(white_mask)
            total_pixels = roi.shape[0] * roi.shape[1]
            
            if total_pixels > 0:
                blue_ratio = blue_pixels / total_pixels
                white_ratio = white_pixels / total_pixels
                
                is_blue_background = (blue_ratio > 0.3 and 
                                    white_ratio < 0.5 and 
                                    blue_ratio > white_ratio)
                
                return is_blue_background
            
            return False
            
        except Exception as e:
            print(f"检测蓝色框时出错: {e}")
            return False


class ChatAnalyzer:
    """聊天分析模块：负责聊天消息结构化"""
    
    def organize_chat_messages(self, marked_texts: List[str]) -> List[Dict]:
        """将标记后的文本组织成结构化聊天消息"""
        nickname_analysis = self._analyze_nicknames(marked_texts)
        
        messages = []
        i = 0
        
        while i < len(marked_texts):
            text = marked_texts[i].strip()
            if not text:
                i += 1
                continue
            
            # 处理时间标记
            if "(时间)" in text:
                time_text = text.replace("(时间)", "").strip()
                messages.append({
                    "type": "time",
                    "time": time_text
                })
                i += 1
                continue
            
            # 处理我的内容标记
            if "(我的内容)" in text:
                my_content_parts = []
                j = i
                
                while j < len(marked_texts):
                    current_text = marked_texts[j].strip()
                    if not current_text:
                        j += 1
                        continue
                    
                    if "(我的内容)" in current_text:
                        content = current_text.replace("(我的内容)", "").strip()
                        if content:
                            my_content_parts.append(content)
                        j += 1
                    else:
                        break
                
                if my_content_parts:
                    combined_content = " ".join(my_content_parts)
                    messages.append({
                        "type": "my_chat",
                        "昵称": "我",
                        "内容": combined_content
                    })
                
                i = j
                continue
            
            # 处理昵称标记
            if "(昵称)" in text:
                nickname = text.replace("(昵称)", "").strip()
                
                content_parts = []
                retract_messages = []
                j = i + 1
                
                while j < len(marked_texts):
                    next_text = marked_texts[j].strip()
                    if not next_text:
                        j += 1
                        continue
                    
                    if ("(昵称)" in next_text or "(时间)" in next_text or "(我的内容)" in next_text):
                        break
                    
                    if "(内容)" in next_text:
                        content = next_text.replace("(内容)", "").strip()
                        if content:
                            if "撤回了一条消息" in content:
                                retract_messages.append(content)
                                print(f"检测到撤回消息: {content}")
                            elif self._is_time_content(content):
                                messages.append({
                                    "type": "time",
                                    "time": content
                                })
                                print(f"在内容中检测到时间: {content}")
                            else:
                                content_parts.append(content)
                    else:
                        content_parts.append(next_text)
                    
                    j += 1
                
                if content_parts:
                    combined_content = " ".join(content_parts)
                    messages.append({
                        "type": "chat",
                        "昵称": nickname,
                        "内容": combined_content
                    })
                else:
                    if self._is_group_name(nickname, i, nickname_analysis):
                        messages.append({
                            "type": "group_name",
                            "群聊名称": nickname
                        })
                        print(f"检测到群聊名称: {nickname}")
                
                for retract_content in retract_messages:
                    messages.append({
                        "type": "retract_message",
                        "撤回信息": retract_content
                    })
                
                i = j
                continue
            
            # 处理孤立的内容标记
            if "(内容)" in text:
                content = text.replace("(内容)", "").strip()
                if "撤回了一条消息" in content:
                    messages.append({
                        "type": "retract_message",
                        "撤回信息": content
                    })
                    print(f"检测到撤回消息: {content}")
                elif self._is_time_content(content):
                    messages.append({
                        "type": "time",
                        "time": content
                    })
                    print(f"在孤立内容中检测到时间: {content}")
                else:
                    messages.append({
                        "type": "chat",
                        "昵称": "未知",
                        "内容": content
                    })
                i += 1
                continue
            
            # 处理未标记的内容
            messages.append({
                "type": "unknown",
                "content": text
            })
            i += 1
        
        return messages
    
    def _analyze_nicknames(self, marked_texts: List[str]) -> Dict:
        """分析所有昵称的出现情况"""
        nickname_info = {}
        first_nickname_index = None
        
        for i, text in enumerate(marked_texts):
            text = text.strip()
            if "(昵称)" in text:
                nickname = text.replace("(昵称)", "").strip()
                
                if nickname not in nickname_info:
                    nickname_info[nickname] = {
                        'first_occurrence': i,
                        'count': 1,
                        'has_content': False
                    }
                    
                    if first_nickname_index is None:
                        first_nickname_index = i
                else:
                    nickname_info[nickname]['count'] += 1
                
                # 检查该昵称是否有对应内容
                j = i + 1
                has_content = False
                while j < len(marked_texts):
                    next_text = marked_texts[j].strip()
                    if not next_text:
                        j += 1
                        continue
                    
                    if ("(昵称)" in next_text or "(时间)" in next_text or "(我的内容)" in next_text):
                        break
                    
                    if "(内容)" in next_text:
                        content = next_text.replace("(内容)", "").strip()
                        if content:
                            has_content = True
                            break
                    
                    j += 1
                
                if has_content:
                    nickname_info[nickname]['has_content'] = True
        
        return {
            'nickname_info': nickname_info,
            'first_nickname_index': first_nickname_index
        }
    
    def _is_group_name(self, nickname: str, position: int, nickname_analysis: Dict) -> bool:
        """判断是否为群聊名称"""
        nickname_info = nickname_analysis['nickname_info']
        first_nickname_index = nickname_analysis['first_nickname_index']
        
        if nickname not in nickname_info:
            return False
        
        info = nickname_info[nickname]
        
        is_first_nickname = (position == first_nickname_index)
        is_unique_or_no_content = (info['count'] == 1 or not info['has_content'])
        
        return is_first_nickname and is_unique_or_no_content
    
    def _is_time_content(self, text: str) -> bool:
        """检查文本是否为时间内容"""
        import re
        
        time_patterns = [
            r'\d{4}年\d{1,2}月\d{1,2}日\d{1,2}:\d{2}',
            r'\d{4}年\d{1,2}月\d{1,2}日',
            r'(昨天|今天|前天|明天)(早上|上午|中午|下午|晚上|凌晨)?\d{1,2}:\d{2}',
            r'(上午|下午|早上|中午|晚上|凌晨)\d{1,2}:\d{2}',
            r'\d{1,2}:\d{2}',
            r'\d{4}-\d{1,2}-\d{1,2}',
            r'\d{1,2}/\d{1,2}',
            r'\d{1,2}月\d{1,2}日',
            r'\d{1,2}:\d{2}:\d{2}',
            r'(昨天|今天|前天|明天)',
            r'周[一二三四五六日天]',
        ]
        
        text = text.strip()
        
        if len(text) > 30:
            return False
            
        exclude_keywords = ['报送', '回执', '会议', '参加', '人员', '工作', '通知', '安排', '要求', '地点', '内容', 
                          '完成', '需要', '前', '后', '开始', '结束', '传包', '表格', '填写', '更新', '自测']
        if any(keyword in text for keyword in exclude_keywords):
            return False
        
        for pattern in time_patterns:
            if re.search(pattern, text):
                match = re.search(pattern, text)
                if match:
                    matched_length = len(match.group())
                    match_ratio = matched_length / len(text)
                    
                    if pattern.startswith('(昨天|今天|前天|明天)') or pattern.startswith('(上午|下午|早上|中午|晚上|凌晨)'):
                        if match_ratio >= 0.4:
                            return True
                    elif pattern.startswith(r'\d{4}年'):
                        if match_ratio >= 0.7:
                            return True
                    else:
                        if match_ratio >= 0.6:
                            return True
        
        return False


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


def main():
    """主函数"""
    processor = RefactoredLongImageOCR(config_path="./config/default_rapidocr.yaml")
    image_path = r"images/image_2.png"
    
    try:
        result = processor.process_long_image(image_path)
        print("\n处理结果摘要:")
        print(f"  - 聊天消息总数: {result['metadata']['total_messages']}")
        print(f"  - 普通聊天消息: {result['statistics']['nickname_messages']} 条")
        print(f"  - 时间消息: {result['statistics']['time_messages']} 条")
        print(f"  - 我的消息: {result['statistics']['my_messages']} 条")
        print(f"  - 群聊名称: {result['statistics']['group_name_messages']} 条")
        print(f"  - 撤回消息: {result['statistics']['retract_messages']} 条")
        print(f"  - 未知内容: {result['statistics']['unknown_messages']} 条")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if os.path.exists("output_images"):
        shutil.rmtree("output_images")
    if os.path.exists("output_json"):
        shutil.rmtree("output_json")
    main()