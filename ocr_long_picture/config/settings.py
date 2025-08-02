#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置常量和设置
"""

# 图像处理配置
DEFAULT_SLICE_HEIGHT = 1200
DEFAULT_OVERLAP = 200

# OCR配置  
DEFAULT_TEXT_SCORE_THRESHOLD = 0.65

# 去重配置
OCR_IOU_THRESHOLD = 0.65
AVATAR_IOU_THRESHOLD = 0.0

# 时间检测模式
TIME_PATTERNS = [
    r'\d{4}年\d{1,2}月\d{1,2}日\d{1,2}:\d{2}',
    r'\d{4}年\d{1,2}月\d{1,2}日',
    r'(昨天|今天|前天|明天)(早上|上午|中午|下午|晚上|凌晨)?\d{1,2}:\d{2}',
    r'(上午|下午|早上|中午|晚上|凌晨)\d{1,2}:\d{2}',
    r'\d{1,2}:\d{2}',
    r'\d{4}-\d{1,2}-\d{1,2}',
    r'\d{1,2}/\d{1,2}',
    r'\d{1,2}月\d{1,2}日',
    r'\d{1,2}:\d{2}:\d{2}',
    r'(昨天|今天|前天|明天)',
    r'周[一二三四五六日天]',
]

# 排除关键词
EXCLUDE_KEYWORDS = [
    '报送', '回执', '会议', '参加', '人员', '工作', '通知', '安排', '要求', '地点', '内容', 
    '完成', '需要', '前', '后', '开始', '结束', '传包', '表格', '填写', '更新', '自测'
]

# 绿色检测HSV范围
GREEN_HSV_LOWER = (35, 40, 40)
GREEN_HSV_UPPER = (85, 255, 255)

# 蓝色检测HSV范围  
BLUE_HSV_LOWER = (100, 30, 80)
BLUE_HSV_UPPER = (130, 180, 255)
WHITE_HSV_LOWER = (0, 0, 200)
WHITE_HSV_UPPER = (180, 30, 255)

# 颜色检测阈值
GREEN_RATIO_THRESHOLD = 0.2
BLUE_RATIO_THRESHOLD = 0.3
WHITE_RATIO_THRESHOLD = 0.5