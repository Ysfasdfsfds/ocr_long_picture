#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像切分模块
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from .config import Config


class ImageSlicer:
    """图像切分器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.slice_height = config.slice_height
        self.overlap = config.overlap
        self.output_images_dir = config.output_images_dir
    
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