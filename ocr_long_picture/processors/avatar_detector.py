#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
头像检测模块：专门处理头像识别
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

from process_avatar import process_avatar_v2, preprocess_and_crop_image, slice_x_croped_values


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