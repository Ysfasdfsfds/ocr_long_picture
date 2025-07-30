#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
长图OCR处理脚本
实现长图切分、OCR识别、结果整合和可视化
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any, Callable, Optional
from rapidocr import RapidOCR
import math
import shutil
from LLM_run import process_with_llm
import re
from sklearn.cluster import DBSCAN, AgglomerativeClustering, MeanShift, estimate_bandwidth, OPTICS
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist, squareform

# 尝试导入hdbscan，如果没有安装则提供友好提示
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("警告: hdbscan未安装，HDBSCAN聚类功能不可用。安装命令: pip install hdbscan")

class LongImageOCR:
    def __init__(self, config_path: str = "default_rapidocr.yaml", clustering_method: str = "adaptive"):
        """
        初始化长图OCR处理器
        
        Args:
            config_path: RapidOCR配置文件路径
            clustering_method: 聚类算法，可选: "dbscan", "hierarchical", "meanshift", "adaptive", "statistical", "sliding_window", "optics", "hdbscan"
        """
        self.engine = RapidOCR(config_path=config_path)
        self.slice_height = 1200  # 切片高度
        self.overlap = 200  # 重叠区域像素
        self.text_score_threshold = 0.65  # 文本识别置信度阈值
        
        # 创建输出目录
        self.output_json_dir = Path("./output_json")
        self.output_images_dir = Path("./output_images")
        self.output_json_dir.mkdir(exist_ok=True)
        self.output_images_dir.mkdir(exist_ok=True)
        
        # 聚类算法设置
        valid_methods = ["dbscan", "hierarchical", "meanshift", "adaptive", "statistical", "sliding_window", "optics", "hdbscan"]
        if clustering_method not in valid_methods:
            print(f"警告: 无效的聚类方法 '{clustering_method}'，使用默认的 'adaptive' 方法")
            clustering_method = "adaptive"
            
        self.clustering_method = clustering_method
        print(f"初始化完成，使用聚类方法: {self.clustering_method}")
        
    def slice_image(self, image_path: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        切分长图
        
        Args:
            image_path: 图像路径
            
        Returns:
            original_image: 原始图像
            slices_info: 切片信息列表，包含切片图像和位置信息
        """
        # 读取原始图像
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"无法读取图像: {image_path}")
            
        h, w, c = original_image.shape
        print(f"原始图像尺寸: {w} x {h}")
        
        if h <= self.slice_height:
            # 图像高度小于等于切片高度，不需要切分
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
            # 计算当前切片的结束位置
            end_y = min(current_y + self.slice_height, h)
            
            # 提取切片
            slice_img = original_image[current_y:end_y, :, :]
            
            # 保存切片信息
            slice_info = {
                'slice': slice_img,
                'start_y': current_y,
                'end_y': end_y,
                'slice_index': slice_index
            }
            slices_info.append(slice_info)
            
            # 保存切片图像
            slice_path = self.output_images_dir / f"slice_{slice_index:03d}.jpg"
            cv2.imwrite(str(slice_path), slice_img)
            print(f"保存切片 {slice_index}: {slice_path}")
            
            # 计算下一个切片的起始位置
            if end_y >= h:
                break
            current_y = end_y - self.overlap
            slice_index += 1
            
        return original_image, slices_info
    
    def process_slices(self, slices_info: List[Dict]) -> List[Dict]:
        """
        对所有切片进行OCR处理和聊天消息分析
        
        Args:
            slices_info: 切片信息列表
            
        Returns:
            slice_results: 每个切片的OCR和聊天分析结果列表
        """
        slice_results = []
        
        for slice_info in slices_info:
            slice_img = slice_info['slice']
            slice_index = slice_info['slice_index']
            start_y = slice_info['start_y']
            
            print(f"处理切片 {slice_index}...")
            
            # 进行OCR识别
            slice_img_rgb = cv2.cvtColor(slice_img, cv2.COLOR_BGR2RGB)
            result = self.engine(slice_img_rgb)

            print(f"切片 {slice_index} 原始识别结果: {[(txt, score) for txt, score in zip(result.txts, result.scores)]}")
            
            # 过滤低置信度结果
            if result.boxes is not None and result.txts is not None:
                filtered_boxes = []
                filtered_txts = []
                filtered_scores = []
                
                for box, txt, score in zip(result.boxes, result.txts, result.scores):
                    if score >= self.text_score_threshold:
                        filtered_boxes.append(box)
                        filtered_txts.append(txt)
                        filtered_scores.append(score)
                
                print(f"切片 {slice_index} 过滤后结果: {[(txt, score) for txt, score in zip(filtered_txts, filtered_scores)]}")
                
                if not filtered_boxes:
                    print(f"切片 {slice_index} 过滤后无有效文本")
                    continue
                
                # 转换坐标到原图坐标系
                adjusted_boxes = []
                for box in filtered_boxes:
                    # box 格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    adjusted_box = []
                    for point in box:
                        adjusted_point = [point[0], point[1] + start_y]
                        adjusted_box.append(adjusted_point)
                    adjusted_boxes.append(adjusted_box)
                
                # 构建该切片的OCR结果
                slice_ocr_result = {
                    'boxes': adjusted_boxes,
                    'txts': filtered_txts,
                    'scores': filtered_scores,
                    'image_shape': slice_img.shape
                }
                
                # 对该切片进行聊天消息分析
                print(f"分析切片 {slice_index} 的聊天消息...")
                slice_chat_result = self.analyze_slice_chat_messages(slice_ocr_result, slice_img, start_y)
                
                # 为该切片的所有消息设置切片索引
                for message in slice_chat_result['messages']:
                    message['slice_index'] = slice_index
                
                # 保存切片结果
                slice_result = {
                    'slice_index': slice_index,
                    'start_y': start_y,
                    'end_y': slice_info['end_y'],
                    'ocr_result': slice_ocr_result,
                    'chat_result': slice_chat_result
                }
                slice_results.append(slice_result)
                
                # 保存切片的OCR可视化结果
                vis_img = result.vis()  # 使用RapidOCR自带的可视化
                if vis_img is not None:
                    vis_path = self.output_images_dir / f"slice_{slice_index:03d}_ocr.jpg"
                    cv2.imwrite(str(vis_path), vis_img)
                    print(f"保存切片OCR结果: {vis_path}")
            else:
                print(f"切片 {slice_index} 未检测到文本")
                
        return slice_results
    
    def merge_chat_results(self, slice_results: List[Dict]) -> Dict:
        """
        合并各个切片的聊天分析结果，处理跨切片的重复消息
        
        Args:
            slice_results: 切片结果列表，每个包含ocr_result和chat_result
            
        Returns:
            merged_chat_result: 合并后的聊天分析结果
        """
        all_messages = []
        
        # 1. 收集所有切片的消息，并设置slice_index
        for slice_result in slice_results:
            slice_index = slice_result['slice_index']
            chat_result = slice_result['chat_result']
            
            for message in chat_result['messages']:
                # 复制消息并设置切片信息
                merged_message = message.copy()
                merged_message['slice_index'] = slice_index
                all_messages.append(merged_message)
        
        print(f"合并前共有 {len(all_messages)} 条消息")
        
        # 2. 按照Y坐标排序所有消息
        all_messages.sort(key=lambda x: x['message_y'])
        
        # 3. 检测跨切片的重复消息
        filtered_messages = []
        
        for i, current_msg in enumerate(all_messages):
            should_keep = True
            duplicate_index = -1
            
            # 检查是否与已添加的消息重复
            for j, existing_msg in enumerate(filtered_messages):
                if self._is_duplicate_message(current_msg, existing_msg):
                    # 发现重复消息
                    y_diff = abs(current_msg['message_y'] - existing_msg['message_y'])
                    print(f"发现重复消息 (Y位置差距: {y_diff:.1f}):")
                    print(f"  当前: 切片{current_msg['slice_index']} - '{current_msg['昵称']}': '{current_msg['内容'][:30]}...' (Y:{current_msg['message_y']:.1f})")
                    print(f"  已存在: 切片{existing_msg['slice_index']} - '{existing_msg['昵称']}': '{existing_msg['内容'][:30]}...' (Y:{existing_msg['message_y']:.1f})")
                    
                    # 比较消息质量，保留更完整的消息
                    quality_comparison = self._compare_message_quality(current_msg, existing_msg)
                    if quality_comparison > 0:
                        # 当前消息质量更高
                        duplicate_index = j
                        should_keep = True
                        print(f"  -> 保留当前消息（质量更高：内容更完整/昵称更明确）")
                    elif quality_comparison < 0:
                        # 已存在的消息质量更高
                        should_keep = False
                        print(f"  -> 保留已存在消息（质量更高：内容更完整/昵称更明确）")
                    else:
                        # 质量相当，保留位置更靠前的
                        if current_msg['message_y'] < existing_msg['message_y']:
                            duplicate_index = j
                            should_keep = True
                            print(f"  -> 质量相当，保留位置更靠前的当前消息")
                        else:
                            should_keep = False
                            print(f"  -> 质量相当，保留位置更靠前的已存在消息")
                    break
            
            if should_keep:
                # 如果需要删除之前的重复项
                if duplicate_index >= 0:
                    del filtered_messages[duplicate_index]
                
                # 添加当前消息
                filtered_messages.append(current_msg)
                # 添加详细的保留信息
                print(f"保留消息 {len(filtered_messages)}: 切片{current_msg['slice_index']} - '{current_msg['昵称']}': '{current_msg['内容'][:30]}...'")
            else:
                print(f"跳过重复消息: 切片{current_msg['slice_index']} - '{current_msg['昵称']}': '{current_msg['内容'][:30]}...'")
        
        print(f"去重后共有 {len(filtered_messages)} 条消息")
        
        # 4. 重新排序并重新编号
        filtered_messages.sort(key=lambda x: x['message_y'])
        
        # 5. 生成最终结果
        final_messages = []
        for i, msg in enumerate(filtered_messages):
            formatted_message = {
                'message_id': i + 1,
                '昵称': msg['昵称'],
                '内容': msg['内容'],
                '时间': msg['time'],
                '是否本人': msg['是否本人'],
                'slice_index': int(msg['slice_index']),  # 转换为Python int
                'y_position': float(msg['message_y'])  # 转换为Python float
            }
            final_messages.append(formatted_message)
        
        return {
            'total_messages': len(final_messages),
            'messages': final_messages
        }
    
    def _is_duplicate_message(self, msg1: Dict, msg2: Dict) -> bool:
        """
        判断两条消息是否重复
        
        Args:
            msg1, msg2: 要比较的消息
            
        Returns:
            是否重复
        """
        # 提取基本信息
        content1 = msg1['内容'].strip() if msg1['内容'] else ""
        content2 = msg2['内容'].strip() if msg2['内容'] else ""
        nickname1 = msg1['昵称'].strip() if msg1['昵称'] else ""
        nickname2 = msg2['昵称'].strip() if msg2['昵称'] else ""
        time1 = msg1['time'].strip() if msg1['time'] else ""
        time2 = msg2['time'].strip() if msg2['time'] else ""
        y_diff = abs(msg1['message_y'] - msg2['message_y'])
        
        # 1. 特殊情况：消息内容被错分为昵称（最高优先级检测）
        # 情况1：msg1有内容，msg2没内容但昵称是msg1的内容
        special_case1 = content1 and not content2 and nickname2 == content1
        # 情况2：msg2有内容，msg1没内容但昵称是msg2的内容  
        special_case2 = content2 and not content1 and nickname1 == content2
        # 情况3：昵称包含对方的内容（部分匹配）
        special_case3 = content1 and not content2 and content1 in nickname2 and len(content1) > 3
        special_case4 = content2 and not content1 and content2 in nickname1 and len(content2) > 3
        
        if special_case1 or special_case2 or special_case3 or special_case4:
            # 对于特殊情况，允许较大的Y位置差距（800像素内，跨多个切片）
            if y_diff <= 800:
                print(f"    检测到特殊重复情况：消息内容被错误分为昵称 (Y差距: {y_diff:.1f})")
                return True
        
        # 2. 内容完全相同的重复（同昵称且内容相同）
        if content1 and content2 and content1 == content2:
            # 如果昵称也相同，强烈怀疑重复
            if nickname1 == nickname2:
                # 允许较大的Y差距（400像素）
                if y_diff <= 400:
                    print(f"    检测到完全相同的重复消息 (Y差距: {y_diff:.1f})")
                    return True
            # 如果昵称不同但内容完全相同，可能是同一人发了两条相同消息
            # 只在较小Y差距内认为是重复（100像素）
            elif y_diff <= 100:
                print(f"    检测到内容相同但昵称不同的可能重复 (Y差距: {y_diff:.1f})")
                return True
        
        # 3. 昵称相似且内容相关的情况
        if self._are_nicknames_similar(nickname1, nickname2) and y_diff <= 300:
            # 如果昵称相似，且一个有内容另一个没有，可能是分割错误
            if (content1 and not content2) or (content2 and not content1):
                print(f"    检测到昵称相似且内容不对称的重复情况 (Y差距: {y_diff:.1f})")
                return True
        
        # 4. 内容相似度检测（在合理Y距离内）
        if content1 and content2 and y_diff <= 300:
            # 去除标点符号后比较
            clean_content1 = re.sub(r'[^\w\u4e00-\u9fff]', '', content1)
            clean_content2 = re.sub(r'[^\w\u4e00-\u9fff]', '', content2)
            
            if clean_content1 == clean_content2 and clean_content1:
                print(f"    检测到去除标点后内容相同的重复 (Y差距: {y_diff:.1f})")
                return True
            
            # 计算相似度
            if clean_content1 and clean_content2:
                similarity = self._calculate_text_similarity(clean_content1, clean_content2)
                
                # 如果相似度很高（>85%），认为是重复
                if similarity > 0.85:
                    print(f"    检测到高相似度重复消息 (相似度: {similarity:.2f}, Y差距: {y_diff:.1f})")
                    return True
                
                # 如果一个内容包含另一个，且长度比例合理
                if clean_content1 in clean_content2 or clean_content2 in clean_content1:
                    min_len = min(len(clean_content1), len(clean_content2))
                    max_len = max(len(clean_content1), len(clean_content2))
                    if min_len > 0 and (min_len / max_len) > 0.8:  # 80%包含度
                        print(f"    检测到包含关系的重复消息 (包含度: {min_len/max_len:.2f}, Y差距: {y_diff:.1f})")
                        return True
        
        # 5. 空内容消息的重复检测
        if not content1 and not content2 and y_diff <= 100:
            # 两个都是空内容且位置很近
            if time1 == time2 and time1:  # 时间相同
                print(f"    检测到时间相同的空内容重复 (Y差距: {y_diff:.1f})")
                return True
            if nickname1 == nickname2:  # 昵称相同
                print(f"    检测到昵称相同的空内容重复 (Y差距: {y_diff:.1f})")
                return True
        
        return False
    
    def _are_nicknames_similar(self, nickname1: str, nickname2: str) -> bool:
        """
        判断两个昵称是否相似
        
        Args:
            nickname1, nickname2: 要比较的昵称
            
        Returns:
            是否相似
        """
        if not nickname1 or not nickname2:
            return False
        
        # 去除特殊字符后比较
        clean_nick1 = re.sub(r'[^\w\u4e00-\u9fff]', '', nickname1)
        clean_nick2 = re.sub(r'[^\w\u4e00-\u9fff]', '', nickname2)
        
        if clean_nick1 == clean_nick2:
            return True
        
        # 检查一个是否包含另一个
        if clean_nick1 and clean_nick2:
            if clean_nick1 in clean_nick2 or clean_nick2 in clean_nick1:
                min_len = min(len(clean_nick1), len(clean_nick2))
                max_len = max(len(clean_nick1), len(clean_nick2))
                if min_len > 0 and (min_len / max_len) > 0.8:  # 80%相似度
                    return True
        
        return False
    
    def _compare_message_quality(self, msg1: Dict, msg2: Dict) -> int:
        """
        比较两条消息的质量
        
        Args:
            msg1, msg2: 要比较的消息
            
        Returns:
            1: msg1质量更高, -1: msg2质量更高, 0: 质量相当
        """
        score1 = 0
        score2 = 0
        
        # 1. 内容长度（更长且有实际内容的更好）
        content1 = msg1['内容'].strip()
        content2 = msg2['内容'].strip()
        
        # 有内容比没有内容好
        if content1 and not content2:
            score1 += 5
        elif content2 and not content1:
            score2 += 5
        elif content1 and content2:
            # 都有内容时，比较长度
            len1 = len(content1)
            len2 = len(content2)
            if len1 > len2:
                score1 += 2
            elif len2 > len1:
                score2 += 2
        
        # 2. 昵称信息（明确的昵称比"未知"好）
        nickname1 = msg1['昵称'].strip()
        nickname2 = msg2['昵称'].strip()
        
        if nickname1 and nickname1 != "未知" and (not nickname2 or nickname2 == "未知"):
            score1 += 3
        elif nickname2 and nickname2 != "未知" and (not nickname1 or nickname1 == "未知"):
            score2 += 3
        
        # 3. 时间信息（有时间信息的更好）
        time1 = msg1['time'].strip() if msg1['time'] else ""
        time2 = msg2['time'].strip() if msg2['time'] else ""
        
        if time1 and not time2:
            score1 += 2
        elif time2 and not time1:
            score2 += 2
        
        # 4. 组件数量（有更多检测到的组件的更好）
        components1 = sum(msg1['components'].values())
        components2 = sum(msg2['components'].values())
        if components1 > components2:
            score1 += 1
        elif components2 > components1:
            score2 += 1
        
        # 5. 切片索引偏好（在重叠区域，倾向于保留来自后面切片的消息，因为可能更完整）
        if msg1['slice_index'] > msg2['slice_index']:
            score1 += 1
        elif msg2['slice_index'] > msg1['slice_index']:
            score2 += 1
        
        if score1 > score2:
            return 1
        elif score2 > score1:
            return -1
        else:
            return 0

    def merge_results(self, slice_results: List[Dict], original_shape: Tuple[int, int, int]) -> Dict:
        """
        整合OCR结果，处理重叠区域
        
        Args:
            slice_results: 切片结果列表，每个包含ocr_result和chat_result
            original_shape: 原始图像形状 (h, w, c)
            
        Returns:
            merged_result: 整合后的结果
        """
        # 从slice_results中提取OCR结果
        ocr_results = []
        for slice_result in slice_results:
            ocr_result = {
                'slice_index': slice_result['slice_index'],
                'start_y': slice_result['start_y'],
                'end_y': slice_result['end_y'],
                'boxes': slice_result['ocr_result']['boxes'],
                'txts': slice_result['ocr_result']['txts'],
                'scores': slice_result['ocr_result']['scores']
            }
            ocr_results.append(ocr_result)
        
        if not ocr_results:
            return {'boxes': [], 'txts': [], 'scores': []}
        
        merged_boxes = []
        merged_txts = []
        merged_scores = []
        
        for i, result in enumerate(ocr_results):
            current_boxes = result['boxes']
            current_txts = result['txts']
            current_scores = result['scores']
            current_start_y = result['start_y']
            current_end_y = result['end_y']
            
            for j, (box, txt, score) in enumerate(zip(current_boxes, current_txts, current_scores)):
                # 检查是否在重叠区域
                box_top_y = min(point[1] for point in box)
                box_bottom_y = max(point[1] for point in box)
                
                # 判断是否需要过滤重叠区域的文本
                should_keep = True
                duplicate_index = -1  # 记录重复项的索引
                
                if i > 0:  # 不是第一个切片
                    # 检查是否在重叠区域内
                    overlap_start = current_start_y
                    overlap_end = current_start_y + self.overlap
                    
                    # 如果文本框主要在重叠区域内，则检查是否与前面的结果重复
                    if box_top_y >= overlap_start and box_top_y < overlap_end:
                        # 在重叠区域内，检查是否与已有结果重复
                        for k, (existing_box, existing_txt, existing_score) in enumerate(zip(merged_boxes, merged_txts, merged_scores)):
                            if self._is_duplicate_text(box, txt, existing_box, existing_txt):
                                # 发现重复，比较置信度
                                if score > existing_score:
                                    # 当前文本框置信度更高，删除之前的，保留当前的
                                    duplicate_index = k
                                    should_keep = True
                                    print(f"发现重复文本 - 当前: '{txt}'({score:.3f}) vs 已存在: '{existing_txt}'({existing_score:.3f}), 保留置信度更高的: '{txt}'")
                                else:
                                    # 之前的文本框置信度更高，跳过当前的
                                    should_keep = False
                                    print(f"发现重复文本 - 当前: '{txt}'({score:.3f}) vs 已存在: '{existing_txt}'({existing_score:.3f}), 保留置信度更高的: '{existing_txt}'")
                                break
                
                if should_keep:
                    # 如果需要删除之前的重复项
                    if duplicate_index >= 0:
                        # 删除置信度较低的重复项
                        del merged_boxes[duplicate_index]
                        del merged_txts[duplicate_index]
                        del merged_scores[duplicate_index]
                    
                    # 添加当前项
                    merged_boxes.append(box)
                    merged_txts.append(txt)
                    merged_scores.append(score)
        
        return {
            'boxes': merged_boxes,
            'txts': merged_txts,
            'scores': merged_scores,
            'image_shape': original_shape
        }
    
    def analyze_slice_chat_messages(self, slice_ocr_result: Dict, slice_image: np.ndarray, start_y_offset: int) -> Dict:
        """
        分析单个切片的聊天消息，按位置和内容进行分类整理
        
        Args:
            slice_ocr_result: 切片的OCR结果
            slice_image: 切片图像
            start_y_offset: 切片在原图中的Y轴偏移量
            
        Returns:
            分析后的聊天消息结构
        """
        if not slice_ocr_result['boxes'] or not slice_ocr_result['txts']:
            return {'messages': []}
        
        # 获取所有文本框的信息
        text_boxes = []
        for i, (box, txt, score) in enumerate(zip(
            slice_ocr_result['boxes'], 
            slice_ocr_result['txts'], 
            slice_ocr_result['scores']
        )):
            # 计算文本框的中心点和边界（相对于原图坐标）
            center_x = float(np.mean([p[0] for p in box]))
            center_y = float(np.mean([p[1] for p in box]))
            min_x = float(min([p[0] for p in box]))
            max_x = float(max([p[0] for p in box]))
            min_y = float(min([p[1] for p in box]))
            max_y = float(max([p[1] for p in box]))
            
            # 检测文本框区域的颜色（需要将坐标转换回切片坐标系）
            slice_box = [[p[0], p[1] - start_y_offset] for p in box]
            is_green_box = self._detect_green_content_box(slice_image, slice_box)
            
            text_boxes.append({
                'id': i,
                'text': txt,
                'score': score,
                'box': box,
                'center_x': center_x,
                'center_y': center_y,
                'min_x': min_x,
                'max_x': max_x,
                'min_y': min_y,
                'max_y': max_y,
                'width': float(max_x - min_x),
                'height': float(max_y - min_y),
                'is_green_box': is_green_box  # 是否为绿色框
            })
        
        # 1. 使用选择的聚类算法在Y轴上进行聚类，分成不同的消息组
        print(f"  在Y轴上进行消息分组...")
        
        # 对于切片，使用固定高度的聚类参数
        clusters = self._cluster_text_boxes(text_boxes, method=self.clustering_method)
        
        # 2. 对每个聚类组进行分析
        messages = []
        
        for cluster_idx, cluster_boxes in enumerate(clusters):
            print(f"  分析消息组 {cluster_idx}，包含 {len(cluster_boxes)} 个文本框")
            
            # 按center_y排序，确保消息的垂直顺序
            cluster_boxes.sort(key=lambda x: x['center_y'])
            
            # 3. 根据颜色和内容特征分类文本框
            green_boxes = []      # 绿色内容框（本人消息）
            time_boxes = []       # 时间戳框
            left_boxes = []       # 左侧框（昵称、头像、其他人内容）
            
            # 时间模式匹配
            time_pattern = r'\d{1,2}:\d{2}|\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日|上午|下午|\d{1,2}:\d{2}:\d{2}'
            
            for box in cluster_boxes:
                # 首先检查是否为绿色框
                if box['is_green_box']:
                    green_boxes.append(box)
                    print(f"    检测到绿色框: '{box['text']}'")
                # 然后检查是否为时间戳
                elif re.search(time_pattern, box['text']):
                    time_boxes.append(box)
                    print(f"    检测到时间框: '{box['text']}'")
                # 其余的归为左侧框
                else:
                    left_boxes.append(box)
                    print(f"    归类为左侧框: '{box['text']}'")
            
            # 4. 判断消息类型和提取信息
            is_self_message = len(green_boxes) > 0  # 有绿色框则为本人消息
            
            # 提取时间
            time_text = ""
            if time_boxes:
                # 选择最符合时间格式的文本
                time_candidates = []
                for box in time_boxes:
                    if re.search(r'\d{1,2}:\d{2}', box['text']):
                        time_candidates.append(box)
                
                if time_candidates:
                    time_text = time_candidates[0]['text']
                else:
                    time_text = time_boxes[0]['text']
            
            # 5. 提取昵称和内容
            nickname = ""
            content_texts = []
            
            if is_self_message:
                # 本人消息：昵称为"本人"，内容来自绿色框
                nickname = "我"
                content_texts = [box['text'] for box in sorted(green_boxes, key=lambda x: (x['center_y'], x['center_x']))]
            else:
                # 他人消息：从左侧框中区分昵称和内容
                if left_boxes:
                    # 过滤掉可能是头像的框
                    potential_nicknames = []
                    potential_contents = []
                    potential_avatars = []
                    
                    for box in left_boxes:
                        text = box['text'].strip()
                        
                        # 判断是否可能是头像框（文本很短、包含特殊符号、尺寸很小）
                        # 只有包含特殊符号才可能是头像框
                        if text in ['□', '○', '◯', '●', '▲', '■',"®"]:
                            potential_avatars.append(box)
                        # 判断是否可能是昵称（较短的有意义文本，位置靠上）
                        elif len(text) <= 10 and re.search(r'[\u4e00-\u9fff\w]', text):
                            potential_nicknames.append(box)
                        # 其余的可能是内容
                        else:
                            potential_contents.append(box)
                    
                    # 选择昵称（优先选择位置最靠上且最短的有意义文本）
                    if potential_nicknames:
                        nickname_box = min(potential_nicknames, key=lambda x: (x['center_y'], len(x['text'])))
                        nickname = nickname_box['text']
                        
                        # 从潜在内容中移除昵称框
                        remaining_boxes = [box for box in left_boxes if box['id'] != nickname_box['id']]
                        content_texts = [box['text'] for box in sorted(remaining_boxes, key=lambda x: (x['center_y'], x['center_x']))]
                    else:
                        # 没有明确的昵称，将所有非头像框的文本作为内容
                        content_texts = [box['text'] for box in sorted(potential_contents, key=lambda x: (x['center_y'], x['center_x']))]
                        nickname = "未知"
            
            # 6. 整理消息
            if content_texts or nickname != "未知":  # 至少要有内容或有效昵称才算一条消息
                content = ' '.join(content_texts) if content_texts else ""
                
                message = {
                    'cluster_id': cluster_idx,
                    '昵称': nickname,
                    '内容': content,
                    'time': time_text,
                    '是否本人': is_self_message,
                    'message_y': float(min(box['center_y'] for box in cluster_boxes)),
                    'components': {
                        'green_boxes_count': len(green_boxes),
                        'time_boxes_count': len(time_boxes),
                        'left_boxes_count': len(left_boxes)
                    }
                }
                messages.append(message)
                
                print(f"    -> 昵称: {message['昵称']}")
                print(f"    -> 内容: {message['内容'][:50]}...")
                print(f"    -> 时间: {message['time']}")
                print(f"    -> 本人消息: {message['是否本人']}")
        
        # 7. 按从上到下的顺序排序消息
        messages.sort(key=lambda x: x['message_y'])
        
        return {
            'total_messages': len(messages),
            'messages': messages
        }

    def analyze_chat_messages(self, merged_result: Dict, original_image: np.ndarray = None) -> Dict:
        """
        分析聊天消息，按位置和内容进行分类整理
        
        Args:
            merged_result: 整合后的OCR结果
            original_image: 原始图像，用于颜色检测
            
        Returns:
            分析后的聊天消息结构
        """
        if not merged_result['boxes'] or not merged_result['txts']:
            return {'messages': []}
        
        # 获取所有文本框的信息
        text_boxes = []
        for i, (box, txt, score) in enumerate(zip(
            merged_result['boxes'], 
            merged_result['txts'], 
            merged_result['scores']
        )):
            # 计算文本框的中心点和边界
            center_x = float(np.mean([p[0] for p in box]))
            center_y = float(np.mean([p[1] for p in box]))
            min_x = float(min([p[0] for p in box]))
            max_x = float(max([p[0] for p in box]))
            min_y = float(min([p[1] for p in box]))
            max_y = float(max([p[1] for p in box]))
            
            # 检测文本框区域的颜色
            is_green_box = False
            if original_image is not None:
                is_green_box = self._detect_green_content_box(original_image, box)
            
            text_boxes.append({
                'id': i,
                'text': txt,
                'score': score,
                'box': box,
                'center_x': center_x,
                'center_y': center_y,
                'min_x': min_x,
                'max_x': max_x,
                'min_y': min_y,
                'max_y': max_y,
                'width': float(max_x - min_x),
                'height': float(max_y - min_y),
                'is_green_box': is_green_box  # 是否为绿色框
            })
        
        # 1. 使用选择的聚类算法在Y轴上进行聚类，分成不同的消息组
        print("步骤1: 在Y轴上进行消息分组...")
        
        # 对于整张图，可以考虑使用自适应参数
        image_height = original_image.shape[0] if original_image is not None else None
        clusters = self._cluster_text_boxes(text_boxes, method=self.clustering_method, image_height=image_height)
        
        # 2. 对每个聚类组进行分析
        messages = []
        
        for cluster_idx, cluster_boxes in enumerate(clusters):
            print(f"分析消息组 {cluster_idx}，包含 {len(cluster_boxes)} 个文本框")
            
            # 按center_y排序，确保消息的垂直顺序
            cluster_boxes.sort(key=lambda x: x['center_y'])
            
            # 3. 根据颜色和内容特征分类文本框
            green_boxes = []      # 绿色内容框（本人消息）
            time_boxes = []       # 时间戳框
            left_boxes = []       # 左侧框（昵称、头像、其他人内容）
            
            # 时间模式匹配
            time_pattern = r'\d{1,2}:\d{2}|\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日|上午|下午|\d{1,2}:\d{2}:\d{2}'
            
            for box in cluster_boxes:
                # 首先检查是否为绿色框
                if box['is_green_box']:
                    green_boxes.append(box)
                    print(f"  检测到绿色框: '{box['text']}'")
                # 然后检查是否为时间戳
                elif re.search(time_pattern, box['text']):
                    time_boxes.append(box)
                    print(f"  检测到时间框: '{box['text']}'")
                # 其余的归为左侧框
                else:
                    left_boxes.append(box)
                    print(f"  归类为左侧框: '{box['text']}'")
            
            # 4. 判断消息类型和提取信息
            is_self_message = len(green_boxes) > 0  # 有绿色框则为本人消息
            
            # 提取时间
            time_text = ""
            if time_boxes:
                # 选择最符合时间格式的文本
                time_candidates = []
                for box in time_boxes:
                    if re.search(r'\d{1,2}:\d{2}', box['text']):
                        time_candidates.append(box)
                
                if time_candidates:
                    time_text = time_candidates[0]['text']
                else:
                    time_text = time_boxes[0]['text']
            
            # 5. 提取昵称和内容
            nickname = ""
            content_texts = []
            
            if is_self_message:
                # 本人消息：昵称为"本人"，内容来自绿色框
                nickname = "我"
                content_texts = [box['text'] for box in sorted(green_boxes, key=lambda x: (x['center_y'], x['center_x']))]
            else:
                # 他人消息：从左侧框中区分昵称和内容
                if left_boxes:
                    # 过滤掉可能是头像的框
                    potential_nicknames = []
                    potential_contents = []
                    potential_avatars = []
                    
                    for box in left_boxes:
                        text = box['text'].strip()
                        
                        # 判断是否可能是头像框（文本很短、包含特殊符号、尺寸很小）
                        # 只有包含特殊符号才可能是头像框
                        if text in ['□', '○', '◯', '●', '▲', '■',"®"]:
                            potential_avatars.append(box)
                        # 判断是否可能是昵称（较短的有意义文本，位置靠上）
                        elif len(text) <= 10 and re.search(r'[\u4e00-\u9fff\w]', text):
                            potential_nicknames.append(box)
                        # 其余的可能是内容
                        else:
                            potential_contents.append(box)
                    
                    # 选择昵称（优先选择位置最靠上且最短的有意义文本）
                    if potential_nicknames:
                        nickname_box = min(potential_nicknames, key=lambda x: (x['center_y'], len(x['text'])))
                        nickname = nickname_box['text']
                        
                        # 从潜在内容中移除昵称框
                        remaining_boxes = [box for box in left_boxes if box['id'] != nickname_box['id']]
                        content_texts = [box['text'] for box in sorted(remaining_boxes, key=lambda x: (x['center_y'], x['center_x']))]
                    else:
                        # 没有明确的昵称，将所有非头像框的文本作为内容
                        content_texts = [box['text'] for box in sorted(potential_contents, key=lambda x: (x['center_y'], x['center_x']))]
                        nickname = "未知"
            
            # 6. 整理消息
            if content_texts or nickname != "未知":  # 至少要有内容或有效昵称才算一条消息
                content = ' '.join(content_texts) if content_texts else ""
                
                message = {
                    'cluster_id': cluster_idx,
                    '昵称': nickname,
                    '内容': content,
                    'time': time_text,
                    '是否本人': is_self_message,
                    'message_y': float(min(box['center_y'] for box in cluster_boxes)),
                    'components': {
                        'green_boxes_count': len(green_boxes),
                        'time_boxes_count': len(time_boxes),
                        'left_boxes_count': len(left_boxes)
                    }
                }
                messages.append(message)
                
                print(f"  -> 昵称: {message['昵称']}")
                print(f"  -> 内容: {message['内容'][:50]}...")
                print(f"  -> 时间: {message['time']}")
                print(f"  -> 本人消息: {message['是否本人']}")
        
        # 7. 按从上到下的顺序排序消息
        messages.sort(key=lambda x: x['message_y'])
        
        # 8. 生成最终结果
        result = {
            'total_messages': len(messages),
            'messages': []
        }
        
        for i, msg in enumerate(messages):
            formatted_message = {
                'message_id': i + 1,
                '昵称': msg['昵称'],
                '内容': msg['内容'],
                '时间': msg['time'],
                '是否本人': msg['是否本人']
            }
            result['messages'].append(formatted_message)
        
        return result
    
    def save_chat_analysis_result(self, chat_result: Dict, image_path: str) -> str:
        """
        保存聊天分析结果
        
        Args:
            chat_result: 聊天分析结果
            image_path: 原始图像路径
            
        Returns:
            保存的JSON文件路径
        """
        # 保存详细的聊天分析结果
        image_name = Path(image_path).stem
        json_path = self.output_json_dir / f"{image_name}_chat_analysis.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(chat_result, f, ensure_ascii=False, indent=2)
        
        print(f"保存聊天分析结果: {json_path}")
        return str(json_path)
    
    def _is_duplicate_text(self, box1: List, txt1: str, box2: List, txt2: str, 
                          threshold: float = 50.0, iou_threshold: float = 0.5) -> bool:
        """
        判断两个文本框是否重复
        
        Args:
            box1, txt1: 第一个文本框和文本
            box2, txt2: 第二个文本框和文本
            threshold: 中心点距离阈值
            iou_threshold: IOU阈值
            
        Returns:
            是否重复
        """
        # 计算两个框的中心点距离
        center1 = np.mean(box1, axis=0)
        center2 = np.mean(box2, axis=0)
        distance = np.linalg.norm(center1 - center2)
        
        # 计算IOU
        iou = self._calculate_iou(box1, box2)
        
        # 如果位置距离太远且IOU很小，肯定不重复
        if distance > threshold and iou < 0.1:
            return False
        
        # 位置接近或有一定重叠的情况下，检查文本内容关系
        # 1. 完全相同
        if txt1 == txt2:
            # 如果文本完全相同，只要有一定重叠就认为是重复
            return iou > 0.1 or distance < threshold
        
        # 2. 去除空格和标点后比较
        clean_txt1 = re.sub(r'[^\w\u4e00-\u9fff]', '', txt1)
        clean_txt2 = re.sub(r'[^\w\u4e00-\u9fff]', '', txt2)
        
        if clean_txt1 == clean_txt2 and clean_txt1:  # 确保不为空
            return iou > 0.1 or distance < threshold
        
        # 3. 检查包含关系（一个文本是另一个的子串）
        if clean_txt1 and clean_txt2:
            # 如果一个文本完全包含另一个，且长度差不太大，认为是重复
            if clean_txt1 in clean_txt2 or clean_txt2 in clean_txt1:
                # 计算文本长度比例，如果差异不大，认为是重复
                min_len = min(len(clean_txt1), len(clean_txt2))
                max_len = max(len(clean_txt1), len(clean_txt2))
                if min_len > 0 and (min_len / max_len) > 0.6:  # 较短文本至少是较长文本的60%
                    # 包含关系的情况下，需要更高的IOU或更近的距离
                    return iou > iou_threshold or distance < threshold * 0.8
        
        # 4. 检查文本相似度（Levenshtein距离）
        if clean_txt1 and clean_txt2:
            similarity = self._calculate_text_similarity(clean_txt1, clean_txt2)
            if similarity > 0.8:  # 相似度大于80%认为是重复
                # 高相似度的情况下，需要有一定的重叠
                return iou > iou_threshold or distance < threshold * 0.7
        
        # 5. 高IOU的情况下，即使文本稍有不同也可能是重复
        if iou > 0.4:  # 高重叠度
            # 检查是否是相似的短文本
            if clean_txt1 and clean_txt2 and max(len(clean_txt1), len(clean_txt2)) <= 10:
                similarity = self._calculate_text_similarity(clean_txt1, clean_txt2)
                if similarity > 0.5:  # 对于短文本，降低相似度要求
                    return True
        
        return False
    
    def _calculate_iou(self, box1: List, box2: List) -> float:
        """
        计算两个文本框的IOU（Intersection over Union）
        
        Args:
            box1, box2: 文本框坐标，格式为[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            
        Returns:
            IOU值，0-1之间的浮点数
        """
        try:
            # 将四边形转换为轴对齐的边界框
            def get_bbox(box):
                x_coords = [point[0] for point in box]
                y_coords = [point[1] for point in box]
                return {
                    'x_min': min(x_coords),
                    'y_min': min(y_coords),
                    'x_max': max(x_coords),
                    'y_max': max(y_coords)
                }
            
            bbox1 = get_bbox(box1)
            bbox2 = get_bbox(box2)
            
            # 计算交集
            x_left = max(bbox1['x_min'], bbox2['x_min'])
            y_top = max(bbox1['y_min'], bbox2['y_min'])
            x_right = min(bbox1['x_max'], bbox2['x_max'])
            y_bottom = min(bbox1['y_max'], bbox2['y_max'])
            
            # 检查是否有交集
            if x_right <= x_left or y_bottom <= y_top:
                return 0.0
            
            # 计算交集面积
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            
            # 计算各自的面积
            area1 = (bbox1['x_max'] - bbox1['x_min']) * (bbox1['y_max'] - bbox1['y_min'])
            area2 = (bbox2['x_max'] - bbox2['x_min']) * (bbox2['y_max'] - bbox2['y_min'])
            
            # 计算并集面积
            union_area = area1 + area2 - intersection_area
            
            # 计算IOU
            if union_area <= 0:
                return 0.0
            
            iou = intersection_area / union_area
            return max(0.0, min(1.0, iou))  # 确保在0-1范围内
            
        except Exception as e:
            # 如果计算出错，返回0
            print(f"计算IOU时出错: {e}")
            return 0.0
    
    def _calculate_text_similarity(self, str1: str, str2: str) -> float:
        """
        计算两个字符串的相似度（基于编辑距离）
        
        Args:
            str1, str2: 要比较的字符串
            
        Returns:
            相似度，0-1之间的浮点数
        """
        if not str1 or not str2:
            return 0.0
        
        # 计算编辑距离
        len1, len2 = len(str1), len(str2)
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        # 初始化
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        
        # 填充dp表
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        
        # 计算相似度
        max_len = max(len1, len2)
        if max_len == 0:
            return 1.0
        
        edit_distance = dp[len1][len2]
        similarity = 1 - (edit_distance / max_len)
        return max(0.0, similarity)
    
    def save_json_result(self, merged_result: Dict, image_path: str) -> str:
        """
        保存JSON结果文件
        
        Args:
            merged_result: 整合后的结果
            image_path: 原始图像路径
            
        Returns:
            保存的JSON文件路径
        """
        # 转换numpy数组为列表，便于JSON序列化
        json_result = {
            'image_path': image_path,
            'image_shape': merged_result['image_shape'],
            'total_texts': len(merged_result['txts']),
            'results': []
        }
        
        for i, (box, txt, score) in enumerate(zip(
            merged_result['boxes'], 
            merged_result['txts'], 
            merged_result['scores']
        )):
            text_item = {
                'id': i,
                'text': txt,
                'confidence': float(score),
                'box': [[float(p[0]), float(p[1])] for p in box],
                'center': [
                    float(np.mean([p[0] for p in box])),
                    float(np.mean([p[1] for p in box]))
                ]
            }
            json_result['results'].append(text_item)
        
        # 保存JSON文件
        image_name = Path(image_path).stem
        json_path = self.output_json_dir / f"{image_name}_ocr_result.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, ensure_ascii=False, indent=2)
        
        print(f"保存JSON结果: {json_path}")
        return str(json_path)
    
    def visualize_final_result(self, original_image: np.ndarray, merged_result: Dict, 
                             image_path: str) -> str:
        """
        可视化最终结果
        
        Args:
            original_image: 原始图像
            merged_result: 整合后的结果
            image_path: 原始图像路径
            
        Returns:
            可视化图像保存路径
        """
        from PIL import Image, ImageDraw, ImageFont
        
        # 转换为PIL图像
        if len(original_image.shape) == 3:
            # BGR to RGB
            pil_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(original_image)
        
        # 创建绘图对象
        draw = ImageDraw.Draw(pil_image)
        
        # 尝试加载中文字体
        try:
            # 在fonts文件夹中查找字体文件
            font_path = Path("fonts/SourceHanSansCN-Regular.otf")
            if font_path.exists():
                font = ImageFont.truetype(str(font_path), size=20)
            else:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # 绘制文本框和文本
        colors = [(0, 255, 0)]
        
        for i, (box, txt, score) in enumerate(zip(
            merged_result['boxes'], 
            merged_result['txts'], 
            merged_result['scores']
        )):
            color = colors[i % len(colors)]
            
            # 绘制文本框
            box_points = [(int(p[0]), int(p[1])) for p in box]
            draw.polygon(box_points, outline=color, width=2)
            
            # 绘制文本
            text_pos = (int(box[0][0]), max(0, int(box[0][1]) - 25))
            draw.text(text_pos, f"{txt} ({score:.2f})", fill=color, font=font)
        
        # 保存可视化结果
        image_name = Path(image_path).stem
        vis_path = self.output_images_dir / f"{image_name}_final_result.jpg"
        
        # 转换回OpenCV格式并保存
        final_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(vis_path), final_img)
        
        print(f"保存最终可视化结果: {vis_path}")
        return str(vis_path)
    
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
        original_image, slices_info = self.slice_image(image_path)
        print(f"共切分为 {len(slices_info)} 个切片")
        
        # 2. OCR处理和切片级聊天分析
        print("步骤2: OCR处理和切片级聊天分析...")
        slice_results = self.process_slices(slices_info)
        print(f"成功处理 {len(slice_results)} 个切片")
        
        # 3. 合并聊天分析结果（去重）
        print("步骤3: 合并聊天分析结果...")
        chat_result = self.merge_chat_results(slice_results)
        chat_json_path = self.save_chat_analysis_result(chat_result, image_path)
        print(f"最终识别出 {len(chat_result['messages'])} 条聊天消息")
        
        # 4. 整合OCR结果（用于可视化）
        print("步骤4: 整合OCR结果...")
        merged_result = self.merge_results(slice_results, original_image.shape)
        print(f"整合后共识别 {len(merged_result['txts'])} 个文本")
        
        # 5. 保存JSON结果
        print("步骤5: 保存JSON结果...")
        json_path = self.save_json_result(merged_result, image_path)
        
        # 6. 可视化最终结果
        print("步骤6: 可视化最终结果...")
        vis_path = self.visualize_final_result(original_image, merged_result, image_path)

        #将结果输入到ollama模型
        print("步骤7: 将结果输入到ollama模型...")
        while True:
            user_question = input("请输入你想问的问题（输入'退出'结束）：")
            if user_question.strip() in ["退出", "q", "Q", "exit"]:
                print("已退出与ollama模型的交互。")
                break
            process_with_llm(user_question, chat_result['messages'])

        # 返回处理摘要
        summary = {
            'input_image': image_path,
            'total_slices': len(slices_info),
            'processed_slices': len(slice_results),
            'total_texts': len(merged_result['txts']),
            'json_output': json_path,
            'visualization_output': vis_path,
            'chat_analysis_output': chat_json_path,
            'slice_images_dir': str(self.output_images_dir)
        }
        
        print("处理完成!")
        print(f"JSON结果: {json_path}")
        print(f"可视化结果: {vis_path}")
        print(f"聊天分析结果: {chat_json_path}")
        print(f"切片图像目录: {self.output_images_dir}")
        
        return summary

    def _detect_green_content_box(self, image: np.ndarray, box: List) -> bool:
        """
        检测文本框区域是否为绿色背景（本人消息框）
        
        Args:
            image: 原始图像
            box: 文本框坐标
            
        Returns:
            是否为绿色框
        """
        try:
            # 获取文本框区域
            points = np.array(box, dtype=np.int32)
            min_x = max(0, int(np.min(points[:, 0])))
            max_x = min(image.shape[1], int(np.max(points[:, 0])))
            min_y = max(0, int(np.min(points[:, 1])))
            max_y = min(image.shape[0], int(np.max(points[:, 1])))
            
            if max_x <= min_x or max_y <= min_y:
                return False
            
            # 提取区域图像
            roi = image[min_y:max_y, min_x:max_x]
            
            if roi.size == 0:
                return False
            
            # 转换为HSV颜色空间进行绿色检测
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # 定义绿色的HSV范围
            # 浅绿色范围（聊天界面中常见的绿色）
            lower_green1 = np.array([35, 40, 40])
            upper_green1 = np.array([85, 255, 255])
            
            # 创建绿色掩码
            mask = cv2.inRange(hsv, lower_green1, upper_green1)
            
            # 计算绿色像素的比例
            green_pixels = cv2.countNonZero(mask)
            total_pixels = roi.shape[0] * roi.shape[1]
            
            if total_pixels > 0:
                green_ratio = green_pixels / total_pixels
                # 如果绿色像素超过20%，认为是绿色框
                return green_ratio > 0.2
            
            return False
            
        except Exception as e:
            print(f"检测绿色框时出错: {e}")
            return False

    def _cluster_text_boxes(self, text_boxes: List[Dict], method: str = None, image_height: int = None) -> List[List[Dict]]:
        """
        对文本框进行纵向聚类
        
        Args:
            text_boxes: 文本框列表
            method: 聚类方法，可选: "dbscan", "hierarchical", "meanshift", "adaptive", "statistical", "sliding_window", "optics", "hdbscan"
            image_height: 图像高度，用于自适应参数设置
            
        Returns:
            clusters: 聚类结果，每个元素是一个包含文本框的列表
        """
        if not text_boxes:
            return []
        
        # 如果未指定方法，使用默认方法
        if method is None:
            method = self.clustering_method
        
        # 提取Y坐标
        y_coordinates = np.array([[tb['center_y']] for tb in text_boxes])
        
        # 按Y坐标排序文本框
        sorted_indices = np.argsort(y_coordinates.flatten())
        sorted_text_boxes = [text_boxes[i] for i in sorted_indices]
        sorted_y_coordinates = y_coordinates[sorted_indices]
        
        print(f"  使用 {method} 算法进行消息分组...")
        
        # 根据指定方法进行聚类
        if method == "dbscan":
            # DBSCAN聚类
            clusters = self._dbscan_clustering(text_boxes, y_coordinates, image_height)
        elif method == "hierarchical":
            # 层次聚类
            clusters = self._hierarchical_clustering(text_boxes, y_coordinates, image_height)
        elif method == "meanshift":
            # Mean Shift聚类
            clusters = self._meanshift_clustering(text_boxes, y_coordinates, image_height)
        elif method == "statistical":
            # 基于统计的聚类
            clusters = self._statistical_clustering(sorted_text_boxes, sorted_y_coordinates)
        elif method == "sliding_window":
            # 滑动窗口动态阈值聚类
            clusters = self._sliding_window_clustering(sorted_text_boxes, sorted_y_coordinates)
        elif method == "optics":
            # OPTICS聚类
            clusters = self._optics_clustering(text_boxes, y_coordinates, image_height)
        elif method == "hdbscan":
            # HDBSCAN聚类
            clusters = self._hdbscan_clustering(text_boxes, y_coordinates, image_height)
        else:  # "adaptive" 或其他
            # 自适应阈值聚类 (默认)
            clusters = self._adaptive_clustering(sorted_text_boxes, sorted_y_coordinates)
            
        # 打印聚类结果
        valid_clusters = [c for c in clusters if c]  # 移除空聚类
        print(f"  共分为 {len(valid_clusters)} 个消息组")
        
        return valid_clusters
    
    def _dbscan_clustering(self, text_boxes: List[Dict], y_coordinates: np.ndarray, image_height: int) -> List[List[Dict]]:
        """
        使用DBSCAN进行聚类
        """
        # 计算适合的eps参数
        if image_height and image_height > self.slice_height:
            # 全图模式使用自适应eps
            eps = max(30, int(image_height * 0.003))  # 图片高度的0.3%，最小30像素
            print(f"  图片高度: {image_height}, 自适应eps值: {eps}")
        else:
            # 切片模式使用固定eps
            eps = 1200//len(text_boxes)+10# 对于1200高度的切片，40像素为经验值
            print(f"  切片高度: {self.slice_height}, 使用eps值: {eps}")
        
        # 运行DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=2)
        cluster_labels = dbscan.fit_predict(y_coordinates)
        
        # 组织聚类结果
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label == -1:  # 噪声点单独成组
                clusters[len(clusters)] = [text_boxes[i]]
            else:
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(text_boxes[i])
        
        return list(clusters.values())
    
    def _hierarchical_clustering(self, text_boxes: List[Dict], y_coordinates: np.ndarray, image_height: int) -> List[List[Dict]]:
        """
        使用层次聚类
        """
        # 计算适合的距离阈值
        if image_height and image_height > self.slice_height:
            distance_threshold = max(30, int(image_height * 0.003))
        else:
            distance_threshold = 40
        print(f"  层次聚类距离阈值: {distance_threshold}")
        
        # 计算距离矩阵
        if len(text_boxes) > 1:
            # 使用层次聚类
            clustering = AgglomerativeClustering(
                n_clusters=None, 
                distance_threshold=distance_threshold,
                linkage='single',
                metric='euclidean'
            )
            cluster_labels = clustering.fit_predict(y_coordinates)
            
            # 组织聚类结果
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(text_boxes[i])
            
            return list(clusters.values())
        else:
            # 只有一个文本框
            return [text_boxes]
    
    def _meanshift_clustering(self, text_boxes: List[Dict], y_coordinates: np.ndarray, image_height: int) -> List[List[Dict]]:
        """
        使用Mean Shift聚类
        """
        # 估计带宽
        if len(y_coordinates) > 1:
            try:
                bandwidth = estimate_bandwidth(y_coordinates, quantile=0.2, n_samples=min(len(y_coordinates), 500))
                if bandwidth == 0:  # 处理极端情况
                    bandwidth = 40
            except Exception as e:
                print(f"  带宽估计失败: {e}, 使用默认值")
                bandwidth = 40
        else:
            bandwidth = 40
        
        print(f"  Mean Shift带宽: {bandwidth}")
        
        # 执行Mean Shift聚类
        if len(text_boxes) > 1:
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            cluster_labels = ms.fit_predict(y_coordinates)
            
            # 组织聚类结果
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(text_boxes[i])
            
            return list(clusters.values())
        else:
            # 只有一个文本框
            return [text_boxes]
    
    def _statistical_clustering(self, text_boxes: List[Dict], y_coordinates: np.ndarray) -> List[List[Dict]]:
        """
        基于统计的聚类
        """
        if len(text_boxes) <= 1:
            return [text_boxes] if text_boxes else []
        
        # 计算相邻Y坐标之间的差值
        y_values = y_coordinates.flatten()
        y_diffs = np.diff(y_values)
        
        if len(y_diffs) == 0:
            return [text_boxes]
        
        # 计算差值的均值和标准差
        mean_diff = np.mean(y_diffs)
        std_diff = np.std(y_diffs)
        
        # 动态阈值 = 均值 + 1.5倍标准差
        threshold = mean_diff + 1.5 * std_diff
        threshold = max(threshold, 30)  # 最小阈值30像素
        print(f"  统计聚类阈值: {threshold:.1f} (均值: {mean_diff:.1f}, 标准差: {std_diff:.1f})")
        
        # 使用动态阈值进行聚类
        clusters = []
        current_cluster = [text_boxes[0]]
        
        for i in range(1, len(text_boxes)):
            y_diff = y_values[i] - y_values[i-1]
            if y_diff > threshold:
                # 开始新的聚类
                clusters.append(current_cluster)
                current_cluster = [text_boxes[i]]
            else:
                # 添加到当前聚类
                current_cluster.append(text_boxes[i])
                
        # 添加最后一个聚类
        if current_cluster:
            clusters.append(current_cluster)
        
        return clusters
        
    def _adaptive_clustering(self, text_boxes: List[Dict], y_coordinates: np.ndarray) -> List[List[Dict]]:
        """
        自适应阈值聚类
        """
        if len(text_boxes) <= 1:
            return [text_boxes] if text_boxes else []
        
        # 初始化参数
        min_gap = 30  # 最小间隔
        y_values = y_coordinates.flatten()
        
        # 查找所有间隔
        y_diffs = []
        for i in range(1, len(y_values)):
            y_diff = y_values[i] - y_values[i-1]
            if y_diff > min_gap:  # 忽略太小的间隔
                y_diffs.append(y_diff)
        
        if not y_diffs:
            return [text_boxes]  # 无法分组时返回单个组
        
        # 计算间隔的分布
        y_diffs = np.array(y_diffs)
        y_diffs_sorted = np.sort(y_diffs)
        
        # 查找间隙的突变点作为阈值（使用差分）
        if len(y_diffs_sorted) > 3:  # 至少需要4个点来查找明显的突变
            diffs_of_diffs = np.diff(y_diffs_sorted)
            # 找最大突变点，或者前25%和后75%的分界点
            try:
                max_change_idx = np.argmax(diffs_of_diffs)
                threshold = y_diffs_sorted[max_change_idx]
                if threshold < min_gap:
                    # 使用分位数作为备选
                    threshold = np.percentile(y_diffs_sorted, 75)
            except:
                threshold = np.percentile(y_diffs_sorted, 75)
        else:
            # 样本太少，使用简单统计
            threshold = np.mean(y_diffs) + np.std(y_diffs)
        
        # 确保阈值合理
        threshold = max(threshold, min_gap)
        print(f"  自适应聚类阈值: {threshold:.1f}")
        
        # 使用自适应阈值进行聚类
        clusters = []
        current_cluster = [text_boxes[0]]
        
        for i in range(1, len(text_boxes)):
            y_diff = y_values[i] - y_values[i-1]
            if y_diff > threshold:
                # 开始新的聚类
                clusters.append(current_cluster)
                current_cluster = [text_boxes[i]]
            else:
                # 添加到当前聚类
                current_cluster.append(text_boxes[i])
                
        # 添加最后一个聚类
        if current_cluster:
            clusters.append(current_cluster)
        
        return clusters

    def _sliding_window_clustering(self, text_boxes: List[Dict], y_coordinates: np.ndarray) -> List[List[Dict]]:
        """
        使用滑动窗口动态阈值聚类
        在聚类过程中，为每个判断点使用局部窗口内的统计特性计算阈值
        """
        if len(text_boxes) <= 1:
            return [text_boxes] if text_boxes else []
        
        # 初始化参数
        min_gap = 30  # 最小间隔
        window_size = min(20, len(text_boxes))  # 窗口大小，最大20个点或全部点
        half_window = window_size // 2  # 半窗口大小
        y_values = y_coordinates.flatten()
        
        print(f"  滑动窗口聚类，窗口大小: {window_size}")
        
        # 使用滑动窗口动态阈值聚类
        clusters = []
        current_cluster = [text_boxes[0]]
        thresholds = []  # 记录每个决策点的阈值
        
        for i in range(1, len(text_boxes)):
            # 计算局部窗口的范围
            start_idx = max(0, i - half_window)
            end_idx = min(len(text_boxes), i + half_window)
            
            # 提取局部窗口内的Y坐标
            local_y_values = y_values[start_idx:end_idx]
            
            # 计算局部窗口内相邻Y坐标的差值
            if len(local_y_values) > 1:
                local_y_diffs = np.diff(local_y_values)
                
                if len(local_y_diffs) > 0:
                    # 计算局部差值的均值和标准差
                    local_mean_diff = np.mean(local_y_diffs)
                    local_std_diff = np.std(local_y_diffs)
                    
                    # 局部动态阈值 = 局部均值 + 1.5倍局部标准差
                    local_threshold = local_mean_diff + 1.5 * local_std_diff
                    local_threshold = max(local_threshold, min_gap)  # 确保阈值不小于最小间隔
                else:
                    local_threshold = min_gap
            else:
                local_threshold = min_gap
            
            # 记录使用的阈值
            thresholds.append(local_threshold)
            
            # 当前点与前一点的Y差值
            y_diff = y_values[i] - y_values[i-1]
            
            # 使用局部阈值进行判断
            if y_diff > local_threshold:
                # 开始新的聚类
                clusters.append(current_cluster)
                current_cluster = [text_boxes[i]]
                print(f"    位置 {i}: Y差值 {y_diff:.1f} > 局部阈值 {local_threshold:.1f}，开始新聚类")
            else:
                # 添加到当前聚类
                current_cluster.append(text_boxes[i])
        
        # 添加最后一个聚类
        if current_cluster:
            clusters.append(current_cluster)
        
        # 输出阈值统计
        if thresholds:
            print(f"  局部阈值统计: 最小值={min(thresholds):.1f}, 最大值={max(thresholds):.1f}, 均值={np.mean(thresholds):.1f}")
        
        return clusters

    def _optics_clustering(self, text_boxes: List[Dict], y_coordinates: np.ndarray, image_height: int = None) -> List[List[Dict]]:
        """
        使用OPTICS进行聚类（Ordering Points To Identify the Clustering Structure）
        OPTICS是DBSCAN的改进版，能够处理不同密度的聚类，不需要指定固定的eps参数
        
        Args:
            text_boxes: 文本框列表
            y_coordinates: Y坐标数组，形状为(n_samples, 1)
            image_height: 图像高度
            
        Returns:
            clusters: 聚类结果
        """
        if len(text_boxes) <= 1:
            return [text_boxes] if text_boxes else []
        
        # 设置最小样本数
        min_samples = 2
        
        # 计算最大邻域半径
        if image_height and image_height > self.slice_height:
            # 对于整图使用自适应参数
            max_eps = max(100, int(image_height * 0.01))  # 图像高度的1%，至少100像素
        else:
            # 对于切片使用固定参数
            max_eps = 100  # 对于1200高度的切片，100像素为较宽松值
            
        print(f"  OPTICS聚类参数: min_samples={min_samples}, max_eps={max_eps}")
        
        # 运行OPTICS聚类
        optics = OPTICS(
            min_samples=min_samples,
            max_eps=max_eps, 
            metric='euclidean',
            cluster_method='xi',  # 使用xi方法自动提取聚类
            xi=0.05              # 相对较小的xi值以获取更多聚类
        )
        
        try:
            # 拟合模型并获取聚类标签
            cluster_labels = optics.fit_predict(y_coordinates)
            
            # 输出可达性距离的统计信息，帮助调整参数
            if hasattr(optics, 'reachability_') and optics.reachability_ is not None:
                reach_distances = optics.reachability_[optics.reachability_ != np.inf]
                if len(reach_distances) > 0:
                    print(f"  可达性距离统计: 最小={np.min(reach_distances):.1f}, "
                          f"最大={np.max(reach_distances):.1f}, 均值={np.mean(reach_distances):.1f}")
            
            # 统计聚类数量和噪声点
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            print(f"  OPTICS聚类结果: {n_clusters} 个聚类, {n_noise} 个噪声点")
            
            # 组织聚类结果
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label == -1:  # 噪声点单独成组
                    clusters[len(clusters)] = [text_boxes[i]]
                else:
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(text_boxes[i])
            
            return list(clusters.values())
            
        except Exception as e:
            print(f"  OPTICS聚类失败: {e}，尝试使用备用方法...")
            # 聚类失败时使用统计聚类作为备用
            return self._statistical_clustering(text_boxes, y_coordinates)

    def _hdbscan_clustering(self, text_boxes: List[Dict], y_coordinates: np.ndarray, image_height: int = None) -> List[List[Dict]]:
        """
        使用HDBSCAN进行聚类（Hierarchical Density-Based Spatial Clustering of Applications with Noise）
        HDBSCAN是DBSCAN的层次化改进版，完全无需eps参数，只需要min_cluster_size
        对于聊天消息聚类是最佳选择
        
        Args:
            text_boxes: 文本框列表
            y_coordinates: Y坐标数组，形状为(n_samples, 1)
            image_height: 图像高度（用于参数自适应）
            
        Returns:
            clusters: 聚类结果
        """
        if not HDBSCAN_AVAILABLE:
            print("  HDBSCAN不可用，回退到统计聚类方法")
            return self._statistical_clustering(text_boxes, y_coordinates)
            
        if len(text_boxes) <= 1:
            return [text_boxes] if text_boxes else []
        
        # 设置HDBSCAN参数
        min_cluster_size = 2    # 最小聚类大小，聊天消息至少2个文本框成组
        min_samples = 1         # 最小样本数，设为1以便检测单独的消息
        
        # 根据数据量动态调整参数
        if len(text_boxes) > 20:
            min_cluster_size = max(2, len(text_boxes) // 10)  # 对于大量文本，增加最小聚类大小
        
        print(f"  HDBSCAN聚类参数: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        
        try:
            # 创建HDBSCAN聚类器
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=2,
                min_samples=1,
                alpha=0.9,  # 微信场景特殊参数
                metric='euclidean',
                cluster_selection_method='leaf',  # 微信场景推荐leaf
                allow_single_cluster=False
            )
            
            # 执行聚类
            cluster_labels = clusterer.fit_predict(y_coordinates)
            
            # 输出聚类稳定性信息
            if hasattr(clusterer, 'cluster_persistence_') and clusterer.cluster_persistence_ is not None:
                persistence_scores = clusterer.cluster_persistence_
                if len(persistence_scores) > 0:
                    print(f"  聚类稳定性得分: 最小={np.min(persistence_scores):.3f}, "
                          f"最大={np.max(persistence_scores):.3f}, 均值={np.mean(persistence_scores):.3f}")
            
            # 输出聚类概率信息（HDBSCAN的优势之一）
            if hasattr(clusterer, 'probabilities_') and clusterer.probabilities_ is not None:
                prob_scores = clusterer.probabilities_
                low_confidence = np.sum(prob_scores < 0.5)
                print(f"  聚类置信度: 低置信度点={low_confidence}/{len(prob_scores)}, "
                      f"平均置信度={np.mean(prob_scores):.3f}")
            
            # 统计聚类结果
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            print(f"  HDBSCAN聚类结果: {n_clusters} 个聚类, {n_noise} 个噪声点")
            
            # 如果只产生了一个大聚类，尝试调整参数重新聚类
            if n_clusters <= 1 and len(text_boxes) > 5:
                print("  检测到过度聚类，尝试减小min_cluster_size参数")
                clusterer_refined = hdbscan.HDBSCAN(
                    min_cluster_size=2,
                    min_samples=3,
                    alpha=0.9,  # 微信场景特殊参数
                    metric='euclidean',
                    cluster_selection_method='leaf',  # 微信场景推荐leaf
                    allow_single_cluster=False
                )
                cluster_labels = clusterer_refined.fit_predict(y_coordinates)
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                n_noise = list(cluster_labels).count(-1)
                print(f"  调整后聚类结果: {n_clusters} 个聚类, {n_noise} 个噪声点")
            
            # 组织聚类结果
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label == -1:  # 噪声点单独成组
                    clusters[f"noise_{i}"] = [text_boxes[i]]
                else:
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(text_boxes[i])
            
            return list(clusters.values())
            
        except Exception as e:
            print(f"  HDBSCAN聚类失败: {e}，回退到统计聚类方法")
            # 聚类失败时使用统计聚类作为备用
            return self._statistical_clustering(text_boxes, y_coordinates)


def main():
    """主函数"""
    import argparse
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='长图OCR处理工具')
    parser.add_argument('--image', type=str, default="images/image copy 5.png", help='要处理的图像路径')
    parser.add_argument('--config', type=str, default="./default_rapidocr.yaml", help='RapidOCR配置文件路径')
    parser.add_argument('--clustering', type=str, default="optics", 
                       choices=["dbscan", "hierarchical", "meanshift", "adaptive", "statistical", "sliding_window", "optics", "hdbscan"],
                       help='聚类算法选择')
    args = parser.parse_args()
    
    # 初始化处理器
    processor = LongImageOCR(config_path=args.config, clustering_method=args.clustering)
    
    try:
        print(f"使用 {args.clustering} 聚类算法处理图像: {args.image}")
        result = processor.process_long_image(args.image)
        print("\n处理结果摘要:")
        for key, value in result.items():
            print(f"  {key}: {value}")
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