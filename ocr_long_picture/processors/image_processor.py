#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像处理模块：负责图像切分和基础处理
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

from ..config.settings import DEFAULT_SLICE_HEIGHT, DEFAULT_OVERLAP


class ImageProcessor:
    """图像处理模块：负责图像切分和基础处理"""
    
    def __init__(self, slice_height: int = DEFAULT_SLICE_HEIGHT, overlap: int = DEFAULT_OVERLAP):
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