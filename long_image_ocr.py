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


class LongImageOCR:
    def __init__(self, config_path: str = "default_rapidocr.yaml"):
        """
        初始化长图OCR处理器
        
        Args:
            config_path: RapidOCR配置文件路径
        """
        self.engine = RapidOCR(config_path=config_path)
        self.slice_height = 1000  # 切片高度
        self.overlap = 200  # 重叠区域像素
        self.text_score_threshold = 0.8  # 文本识别置信度阈值
        
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
        import re
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
        
        # 返回处理摘要
        summary = {
            'input_image': image_path,
            'total_slices': len(slices_info),
            'processed_slices': len(ocr_results),
            'total_texts': len(merged_result['txts']),
            'json_output': json_path,
            'visualization_output': vis_path,
            'slice_images_dir': str(self.output_images_dir)
        }
        
        print("处理完成!")
        print(f"JSON结果: {json_path}")
        print(f"可视化结果: {vis_path}")
        print(f"切片图像目录: {self.output_images_dir}")
        
        return summary


def main():
    """主函数"""
    # 初始化处理器
    processor = LongImageOCR(config_path="default_rapidocr.yaml")
    
    # 处理长图
    image_path = r"images\long_picture_2.png"
    
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
    main() 