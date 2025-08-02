#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置模块
"""

from .settings import *

__all__ = [
    'DEFAULT_SLICE_HEIGHT',
    'DEFAULT_OVERLAP', 
    'DEFAULT_TEXT_SCORE_THRESHOLD',
    'OCR_IOU_THRESHOLD',
    'AVATAR_IOU_THRESHOLD',
    'TIME_PATTERNS',
    'EXCLUDE_KEYWORDS'
]