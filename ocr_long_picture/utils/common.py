#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用工具函数
"""

from pathlib import Path
from typing import List, Union
import shutil


def calculate_box_iou(box1, box2) -> float:
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


def get_box_center_y(box) -> float:
    """获取box的中心Y坐标"""
    if isinstance(box[0], list):
        return sum(pt[1] for pt in box) / 4
    else:
        return box[1] + box[3] / 2


def get_box_y_min(box) -> float:
    """获取box的最小Y坐标"""
    if isinstance(box[0], list):
        return box[3][1]
    else:
        return box[1] + box[3] / 2


def create_output_directories(*dirs: Union[str, Path]) -> None:
    """创建输出目录"""
    for dir_path in dirs:
        Path(dir_path).mkdir(exist_ok=True)


def cleanup_directories(*dirs: Union[str, Path]) -> None:
    """清理目录"""
    for dir_path in dirs:
        path = Path(dir_path)
        if path.exists():
            shutil.rmtree(path)