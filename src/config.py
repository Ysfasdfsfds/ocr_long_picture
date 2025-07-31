#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块
"""

from pathlib import Path
from typing import Dict, Any


class Config:
    """配置管理类"""
    
    def __init__(self):
        # OCR配置
        self.rapidocr_config_path = "default_rapidocr.yaml"
        self.text_score_threshold = 0.65
        
        # 图像切分配置
        self.slice_height = 1200
        self.overlap = 200
        
        # 去重配置
        self.ocr_iou_threshold = 0.65
        self.avatar_iou_threshold = 0.0
        
        # 输出目录配置
        self.output_json_dir = Path("./output_json")
        self.output_images_dir = Path("./output_images")
        self.debug_images_dir = Path("./debug_images")
        
        # 创建输出目录
        self._create_directories()
    
    def _create_directories(self):
        """创建必要的输出目录"""
        self.output_json_dir.mkdir(exist_ok=True)
        self.output_images_dir.mkdir(exist_ok=True)
        self.debug_images_dir.mkdir(exist_ok=True)
    
    def get_ocr_config(self) -> Dict[str, Any]:
        """获取OCR相关配置"""
        return {
            'config_path': self.rapidocr_config_path,
            'text_score_threshold': self.text_score_threshold
        }
    
    def get_slice_config(self) -> Dict[str, Any]:
        """获取切分相关配置"""
        return {
            'slice_height': self.slice_height,
            'overlap': self.overlap
        }
    
    def get_dedup_config(self) -> Dict[str, Any]:
        """获取去重相关配置"""
        return {
            'ocr_iou_threshold': self.ocr_iou_threshold,
            'avatar_iou_threshold': self.avatar_iou_threshold
        }