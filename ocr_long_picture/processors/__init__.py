#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理器模块
"""

from .image_processor import ImageProcessor
from .ocr_processor import OCRProcessor
from .avatar_detector import AvatarDetector
from .data_deduplicator import DataDeduplicator
from .content_marker import ContentMarker

__all__ = [
    'ImageProcessor',
    'OCRProcessor', 
    'AvatarDetector',
    'DataDeduplicator',
    'ContentMarker'
]