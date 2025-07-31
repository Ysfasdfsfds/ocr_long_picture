#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
头像检测模块
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from .config import Config


class AvatarDetector:
    """头像检测器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.output_images_dir = config.output_images_dir
        self.slice_x_croped_values = {}  # 存储每个切片的x_croped值
    
    def detect_avatars_in_slice(self, slice_img: np.ndarray, slice_index: int, start_y: int) -> List[Tuple]:
        """
        在切片中检测头像
        
        Args:
            slice_img: 切片图像
            slice_index: 切片索引
            start_y: 切片在原图中的起始Y坐标
            
        Returns:
            头像位置列表 [(x, y, w, h), ...]，坐标已转换为原图坐标
        """
        # 预处理图像
        processed_img, binary = self._preprocess_image(slice_img)
        
        # 提取轮廓和外接矩形
        rects, x_croped, target_box = self._extract_contours_and_rects(binary, processed_img)
        
        # 存储target_box到原图坐标
        if target_box is not None:
            x, y, w, h = target_box
            target_box_original = (x, y + start_y, w, h)
            self.slice_x_croped_values[slice_index] = target_box_original
            print(f"切片 {slice_index} 的target_box原图坐标: {target_box_original}")
        else:
            self.slice_x_croped_values[slice_index] = None
        
        # 如果有x_croped值，对图像进行裁剪
        if x_croped is not None:
            cropped_slice = slice_img[0:slice_img.shape[0], 0:x_croped]
            print(f"切片 {slice_index} 使用x_croped={x_croped}进行裁剪")
        else:
            cropped_slice = slice_img
            print(f"切片 {slice_index} 未进行x裁剪，使用原始图像")
        
        # 保存调试图像
        debug_path = self.config.debug_images_dir / f"slice_{slice_index:03d}_avatar.jpg"
        cv2.imwrite(str(debug_path), cropped_slice)
        
        # 在裁剪后的图像中检测头像
        avatar_positions = self._detect_avatars_in_cropped_image(cropped_slice)
        
        # 将坐标转换为原图坐标
        if avatar_positions:
            restored_positions = []
            for (x, y, w, h) in avatar_positions:
                restored_box = (x, y + start_y, w, h)
                restored_positions.append(restored_box)
            avatar_positions = restored_positions
        
        # 按Y坐标排序
        if avatar_positions:
            avatar_positions = sorted(avatar_positions, key=lambda rect: rect[1])
        
        return avatar_positions
    
    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预处理图像：转灰度、模糊、二值化
        
        Args:
            image: 输入图像
            
        Returns:
            tuple: (原图, 二值化图像)
        """
        # 转灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 二值化
        _, binary = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY)
        
        # 反转颜色（头像通常是深色背景）
        binary = 255 - binary
        
        return image, binary
    
    def _extract_contours_and_rects(self, binary_img: np.ndarray, img: np.ndarray) -> Tuple[List, Optional[int], Optional[Tuple]]:
        """
        提取轮廓并计算外接矩形
        
        Args:
            binary_img: 二值化图像
            img: 原图像
            
        Returns:
            tuple: (外接矩形列表, x_croped值, target_box)
        """
        # 查找轮廓
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 计算所有轮廓的外接矩形
        rects = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            rects.append((x, y, w, h))
        
        # 按面积从大到小排序
        rects = sorted(rects, key=lambda box: box[2] * box[3], reverse=True)
        
        # 保存可视化结果
        result_img = img.copy()
        for (x, y, w, h) in rects:
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        debug_path = self.output_images_dir / "rects_visualization.jpg"
        cv2.imwrite(str(debug_path), result_img)
        
        x_croped = None
        target_box = None
        
        if rects:
            # 寻找最适合的target_box
            target_box, x_croped = self._find_target_box(rects, img)
        
        return rects, x_croped, target_box
    
    def _find_target_box(self, rects: List[Tuple], img: np.ndarray) -> Tuple[Optional[Tuple], Optional[int]]:
        """
        寻找最适合的target_box用于确定x_croped
        
        Args:
            rects: 外接矩形列表
            img: 原图像
            
        Returns:
            tuple: (target_box, x_croped值)
        """
        def is_strict_square(r):
            """判断是否严格趋近于正方形"""
            ratio = r[2] / r[3] if r[3] != 0 else 0
            return 0.9 <= ratio <= 1.1 or 0.9 <= (1/ratio) <= 1.1
        
        # 取所有x坐标并排序，找到前三个不同的x坐标值
        x_list = [r[0] for r in rects]
        unique_x = sorted(set(x_list))
        
        # 取前三个x坐标值（如果不足3个就取全部）
        top_3_x = unique_x[:3]
        print(f"左侧前三个x坐标值: {top_3_x}")
        
        # 挑选出左侧前三的所有框
        left_top3_rects = [r for r in rects if r[0] in top_3_x]
        print(f"左侧前三的框总数: {len(left_top3_rects)}")
        
        # 筛选掉不严格趋于正方形的框
        square_rects = [r for r in left_top3_rects if is_strict_square(r)]
        print(f"严格趋于正方形的框数量: {len(square_rects)}")
        
        # 在剩余框中选择面积最大的作为target_box
        if square_rects:
            target_box = max(square_rects, key=lambda r: r[2] * r[3])
            x_croped = target_box[0] + target_box[2] + 3
            print(f"选中的target_box: x={target_box[0]}, y={target_box[1]}, w={target_box[2]}, h={target_box[3]}, 面积={target_box[2]*target_box[3]}")
        else:
            # 降级处理：如果没有严格正方形的框，从所有框中选择最左侧的
            print("警告: 没有找到严格趋于正方形的框，降级为选择最左侧的框")
            target_box = min(rects, key=lambda r: r[0])
            x_croped = target_box[0] + target_box[2] + 3
        
        # 绘制target_box
        x, y, w, h = target_box
        img_square = img.copy()
        cv2.rectangle(img_square, (x, y), (x + w, y + h), (0, 0, 255), 3)
        
        debug_path = self.output_images_dir / "x_croped_target_box.jpg"
        cv2.imwrite(str(debug_path), img_square)
        print(f"用于形成x_croped的框: {target_box}")
        
        return target_box, x_croped
    
    def _detect_avatars_in_cropped_image(self, cropped_img: np.ndarray) -> List[Tuple]:
        """
        在裁剪后的图像中检测头像
        
        Args:
            cropped_img: 裁剪后的图像
            
        Returns:
            头像位置列表 [(x, y, w, h), ...]
        """
        # 预处理裁剪后的图像
        _, binary = self._preprocess_image(cropped_img)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 计算外接矩形
        rects = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            rects.append((x, y, w, h))
        
        # 按面积排序
        rects = sorted(rects, key=lambda box: box[2] * box[3], reverse=True)
        
        # 过滤出近似正方形的框
        square_rects = [r for r in rects if 0.9 <= r[2]/r[3] <= 1.1 or 0.9 <= r[3]/r[2] <= 1.1]
        
        if square_rects:
            # 统计尺寸，过滤异常值
            filtered_rects = self._filter_by_size_consistency(square_rects)
            
            # 应用NMS和合并
            final_rects = self._apply_nms_and_merge(filtered_rects, cropped_img)
            
            return final_rects
        
        return []
    
    def _filter_by_size_consistency(self, square_rects: List[Tuple]) -> List[Tuple]:
        """根据尺寸一致性过滤框"""
        if not square_rects:
            return []
        
        # 统计所有正方形框的w和h
        ws = [r[2] for r in square_rects]
        hs = [r[3] for r in square_rects]
        mean_w = sum(ws) / len(ws)
        mean_h = sum(hs) / len(hs)
        std_w = (sum([(w - mean_w) ** 2 for w in ws]) / len(ws)) ** 0.5
        std_h = (sum([(h - mean_h) ** 2 for h in hs]) / len(hs)) ** 0.5
        
        # 只保留w和h都接近均值的正方形框
        filtered_rects = [
            r for r in square_rects
            if abs(r[2] - mean_w) <= std_w and abs(r[3] - mean_h) <= std_h
        ]
        
        return filtered_rects
    
    def _apply_nms_and_merge(self, rects: List[Tuple], img: np.ndarray) -> List[Tuple]:
        """应用NMS和合并相邻框"""
        if not rects:
            return []
        
        # 计算合并阈值
        max_box = rects[0]  # 面积最大的框
        max_dim = max(max_box[2], max_box[3])  # 最长边
        merge_threshold = max_dim
        
        # 应用NMS
        nms_rects = self._apply_nms(rects, iou_threshold=0.0)
        
        # 合并相邻框
        merged_rects = self._merge_nearby_boxes(nms_rects, merge_threshold)
        
        # 绘制结果
        merged_rects_img = img.copy()
        for (x, y, w, h) in merged_rects:
            cv2.rectangle(merged_rects_img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
        
        debug_path = self.output_images_dir / "merged_rects_drawn.jpg"
        cv2.imwrite(str(debug_path), merged_rects_img)
        
        return merged_rects
    
    def _apply_nms(self, rects: List[Tuple], iou_threshold: float = 0.0) -> List[Tuple]:
        """应用非最大抑制"""
        keep_rects = []
        for rect in rects:
            keep = True
            for kept_rect in keep_rects:
                if self._calculate_iou(rect, kept_rect) > iou_threshold:
                    keep = False
                    break
            if keep:
                keep_rects.append(rect)
        return keep_rects
    
    def _merge_nearby_boxes(self, rects: List[Tuple], merge_threshold: float) -> List[Tuple]:
        """合并相邻的边界框"""
        merged_rects = []
        used = [False] * len(rects)
        
        for i in range(len(rects)):
            if used[i]:
                continue
            
            # 找到所有需要与当前框合并的框
            group = [rects[i]]
            used[i] = True
            
            for j in range(i + 1, len(rects)):
                if used[j]:
                    continue
                
                # 检查是否与组中任何一个框相邻
                should_add = False
                for box_in_group in group:
                    if self._should_merge(box_in_group, rects[j], merge_threshold):
                        should_add = True
                        break
                
                if should_add:
                    group.append(rects[j])
                    used[j] = True
            
            # 合并这一组框
            merged_box = self._merge_boxes(group)
            if merged_box:
                merged_rects.append(merged_box)
        
        return merged_rects
    
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """计算两个框的IOU"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # 计算交集
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        
        # 计算并集
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0
        
        return inter_area / union_area
    
    def _should_merge(self, box1: Tuple, box2: Tuple, distance_threshold: float) -> bool:
        """判断两个框是否应该合并"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # 计算两个框中心点之间的距离
        center1_x, center1_y = x1 + w1 // 2, y1 + h1 // 2
        center2_x, center2_y = x2 + w2 // 2, y2 + h2 // 2
        distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
        
        # 如果距离小于阈值或IOU大于0.01，则合并
        return distance < distance_threshold or self._calculate_iou(box1, box2) > 0.01
    
    def _merge_boxes(self, boxes: List[Tuple]) -> Optional[Tuple]:
        """合并一组框为一个最小外接框"""
        if not boxes:
            return None
        
        min_x = min(box[0] for box in boxes)
        min_y = min(box[1] for box in boxes)
        max_x = max(box[0] + box[2] for box in boxes)
        max_y = max(box[1] + box[3] for box in boxes)
        
        return (min_x, min_y, max_x - min_x, max_y - min_y)
    
    def get_x_croped_values(self) -> Dict:
        """获取所有切片的x_croped值"""
        return self.slice_x_croped_values