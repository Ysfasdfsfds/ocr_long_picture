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
from typing import List, Dict, Tuple, Any
from rapidocr import RapidOCR
import math
import shutil
from LLM_run import process_with_llm
import re
from sklearn.cluster import DBSCAN

class LongImageOCR:
    def __init__(self, config_path: str = "default_rapidocr.yaml"):
        """
        初始化长图OCR处理器
        
        Args:
            config_path: RapidOCR配置文件路径
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
        对所有切片进行OCR处理
        
        Args:
            slices_info: 切片信息列表
            
        Returns:
            ocr_results: OCR结果列表
        """
        ocr_results = []
        
        for slice_info in slices_info:
            slice_img = slice_info['slice']
            slice_index = slice_info['slice_index']
            start_y = slice_info['start_y']
            
            print(f"处理切片 {slice_index}...")
            
            # 进行OCR识别
            slice_img = cv2.cvtColor(slice_img, cv2.COLOR_BGR2RGB)
            result = self.engine(slice_img)

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
                
                # 保存OCR结果
                slice_result = {
                    'slice_index': slice_index,
                    'start_y': start_y,
                    'end_y': slice_info['end_y'],
                    'boxes': adjusted_boxes,
                    'txts': filtered_txts,
                    'scores': filtered_scores
                }
                ocr_results.append(slice_result)
                
                # 保存切片的OCR可视化结果
                vis_img = result.vis()  # 使用RapidOCR自带的可视化
                if vis_img is not None:
                    vis_path = self.output_images_dir / f"slice_{slice_index:03d}_ocr.jpg"
                    cv2.imwrite(str(vis_path), vis_img)
                    print(f"保存切片OCR结果: {vis_path}")
            else:
                print(f"切片 {slice_index} 未检测到文本")
                
        return ocr_results
    
    def merge_results(self, ocr_results: List[Dict], original_shape: Tuple[int, int, int]) -> Dict:
        """
        整合OCR结果，处理重叠区域
        
        Args:
            ocr_results: OCR结果列表
            original_shape: 原始图像形状 (h, w, c)
            
        Returns:
            merged_result: 整合后的结果
        """
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
            center_x = np.mean([p[0] for p in box])
            center_y = np.mean([p[1] for p in box])
            min_x = min([p[0] for p in box])
            max_x = max([p[0] for p in box])
            min_y = min([p[1] for p in box])
            max_y = max([p[1] for p in box])
            
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
                'width': max_x - min_x,
                'height': max_y - min_y,
                'is_green_box': is_green_box  # 是否为绿色框
            })
        
        # 1. 使用DBSCAN在Y轴上进行聚类，分成不同的消息组
        print("步骤1: 使用DBSCAN在Y轴上进行消息分组...")
        y_coordinates = np.array([[tb['center_y']] for tb in text_boxes])
        
        # DBSCAN参数：eps是邻域半径，min_samples是最小样本数
        dbscan = DBSCAN(eps=70, min_samples=1)
        cluster_labels = dbscan.fit_predict(y_coordinates)
        
        # 将文本框按聚类分组
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(text_boxes[i])
        
        print(f"共分为 {len(clusters)} 个消息组")
        
        # 2. 对每个聚类组进行分析
        messages = []
        
        for cluster_id, cluster_boxes in clusters.items():
            if cluster_id == -1:  # 噪声点，跳过
                continue
                
            print(f"分析消息组 {cluster_id}，包含 {len(cluster_boxes)} 个文本框")
            
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
                    'cluster_id': cluster_id,
                    '昵称': nickname,
                    '内容': content,
                    'time': time_text,
                    '是否本人': is_self_message,
                    'message_y': min(box['center_y'] for box in cluster_boxes),
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
        
        # 2. OCR处理
        print("步骤2: OCR处理...")
        ocr_results = self.process_slices(slices_info)
        print(f"成功处理 {len(ocr_results)} 个切片")
        
        # 3. 整合结果
        print("步骤3: 整合结果...")
        merged_result = self.merge_results(ocr_results, original_image.shape)
        print(f"整合后共识别 {len(merged_result['txts'])} 个文本")
        
        
        # 4. 保存JSON结果
        print("步骤4: 保存JSON结果...")
        json_path = self.save_json_result(merged_result, image_path)

      
        
        # 5. 可视化最终结果
        print("步骤5: 可视化最终结果...")
        vis_path = self.visualize_final_result(original_image, merged_result, image_path)
        
        # 6. 分析聊天消息
        print("步骤6: 分析聊天消息...")
        chat_result = self.analyze_chat_messages(merged_result, original_image)
        chat_json_path = self.save_chat_analysis_result(chat_result, image_path)
        # print(f"聊天分析结果: {chat_result}")

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
            'processed_slices': len(ocr_results),
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


def main():
    """主函数"""
    # 初始化处理器
    processor = LongImageOCR(config_path="./default_rapidocr.yaml")
    
    # 处理长图
    # image_path = r"images/image copy 3.png"
    image_path = r"images/image copy 4.png"
    
    try:
        result = processor.process_long_image(image_path)
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
    main() 