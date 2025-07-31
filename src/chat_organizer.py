#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聊天消息组织模块
"""

import re
from typing import List, Dict
from .config import Config


class ChatOrganizer:
    """聊天消息组织器"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def organize_chat_messages(self, marked_texts: List[str]) -> List[Dict]:
        """
        将标记后的文本组织成结构化聊天消息
        
        Args:
            marked_texts: 标记后的文本列表
            
        Returns:
            结构化的聊天消息列表
        """
        # 首先分析所有昵称信息
        nickname_analysis = self._analyze_nicknames(marked_texts)
        
        messages = []
        i = 0
        
        while i < len(marked_texts):
            text = marked_texts[i].strip()
            if not text:
                i += 1
                continue
            
            # 处理时间标记
            if "(时间)" in text:
                time_text = text.replace("(时间)", "").strip()
                messages.append({
                    "type": "time",
                    "time": time_text
                })
                i += 1
                continue
            
            # 处理我的内容标记
            if "(我的内容)" in text:
                my_content_parts = []
                j = i
                
                # 收集连续的"我的内容"标记
                while j < len(marked_texts):
                    current_text = marked_texts[j].strip()
                    if not current_text:
                        j += 1
                        continue
                    
                    if "(我的内容)" in current_text:
                        content = current_text.replace("(我的内容)", "").strip()
                        if content:
                            my_content_parts.append(content)
                        j += 1
                    else:
                        # 遇到非"我的内容"标记，停止收集
                        break
                
                # 合并所有内容并创建消息
                if my_content_parts:
                    combined_content = " ".join(my_content_parts)
                    messages.append({
                        "type": "my_chat",
                        "昵称": "我",
                        "内容": combined_content
                    })
                
                # 跳过已处理的内容
                i = j
                continue
            
            # 处理昵称标记
            if "(昵称)" in text:
                nickname = text.replace("(昵称)", "").strip()
                
                # 向前查找是否有对应的内容
                content_parts = []
                retract_messages = []  # 收集撤回消息，稍后处理
                j = i + 1
                
                # 收集后续的内容标记，直到遇到下一个昵称、时间或我的内容
                while j < len(marked_texts):
                    next_text = marked_texts[j].strip()
                    if not next_text:
                        j += 1
                        continue
                    
                    # 如果遇到新的昵称、时间或我的内容，停止收集
                    if ("(昵称)" in next_text or "(时间)" in next_text or "(我的内容)" in next_text):
                        break
                    
                    # 收集内容标记
                    if "(内容)" in next_text:
                        content = next_text.replace("(内容)", "").strip()
                        if content:
                            # 检查是否为撤回消息
                            if "撤回了一条消息" in content:
                                # 撤回消息暂存，稍后处理
                                retract_messages.append(content)
                                print(f"检测到撤回消息: {content}")
                            # 检查是否为时间内容（二次检测）
                            elif self._is_time_content(content):
                                # 如果是时间内容，添加为时间消息
                                messages.append({
                                    "type": "time",
                                    "time": content
                                })
                                print(f"在内容中检测到时间: {content}")
                            else:
                                content_parts.append(content)
                    else:
                        # 未标记的内容也收集（可能是OCR错误）
                        content_parts.append(next_text)
                    
                    j += 1
                
                # 判断是否有内容
                if content_parts:
                    # 有内容，创建正常聊天消息
                    combined_content = " ".join(content_parts)
                    messages.append({
                        "type": "chat",
                        "昵称": nickname,
                        "内容": combined_content
                    })
                else:
                    # 没有内容，检查是否为群聊名称
                    if self._is_group_name(nickname, i, nickname_analysis):
                        messages.append({
                            "type": "group_name",
                            "群聊名称": nickname
                        })
                        print(f"检测到群聊名称: {nickname}")
                
                # 在聊天消息之后添加撤回消息
                for retract_content in retract_messages:
                    messages.append({
                        "type": "retract_message",
                        "撤回信息": retract_content
                    })
                
                # 跳过已处理的内容
                i = j
                continue
            
            # 处理孤立的内容标记（没有对应昵称的内容）
            if "(内容)" in text:
                content = text.replace("(内容)", "").strip()
                # 检查是否为撤回消息
                if "撤回了一条消息" in content:
                    messages.append({
                        "type": "retract_message",
                        "撤回信息": content
                    })
                    print(f"检测到撤回消息: {content}")
                # 检查是否为时间内容（二次检测）
                elif self._is_time_content(content):
                    messages.append({
                        "type": "time",
                        "time": content
                    })
                    print(f"在孤立内容中检测到时间: {content}")
                else:
                    messages.append({
                        "type": "chat",
                        "昵称": "未知",
                        "内容": content
                    })
                i += 1
                continue
            
            # 处理未标记的内容
            messages.append({
                "type": "unknown",
                "content": text
            })
            i += 1
        
        return messages
    
    def _analyze_nicknames(self, marked_texts: List[str]) -> Dict:
        """
        分析所有昵称的出现情况
        
        Args:
            marked_texts: 标记后的文本列表
            
        Returns:
            昵称分析结果字典
        """
        nickname_info = {}
        first_nickname_index = None
        
        for i, text in enumerate(marked_texts):
            text = text.strip()
            if "(昵称)" in text:
                nickname = text.replace("(昵称)", "").strip()
                
                if nickname not in nickname_info:
                    nickname_info[nickname] = {
                        'first_occurrence': i,
                        'count': 1,
                        'has_content': False
                    }
                    
                    # 记录第一个出现的昵称位置
                    if first_nickname_index is None:
                        first_nickname_index = i
                else:
                    nickname_info[nickname]['count'] += 1
                
                # 检查该昵称是否有对应内容
                j = i + 1
                has_content = False
                while j < len(marked_texts):
                    next_text = marked_texts[j].strip()
                    if not next_text:
                        j += 1
                        continue
                    
                    if ("(昵称)" in next_text or "(时间)" in next_text or "(我的内容)" in next_text):
                        break
                    
                    if "(内容)" in next_text:
                        content = next_text.replace("(内容)", "").strip()
                        if content:
                            has_content = True
                            break
                    
                    j += 1
                
                if has_content:
                    nickname_info[nickname]['has_content'] = True
        
        return {
            'nickname_info': nickname_info,
            'first_nickname_index': first_nickname_index
        }
    
    def _is_group_name(self, nickname: str, position: int, nickname_analysis: Dict) -> bool:
        """
        判断是否为群聊名称
        
        Args:
            nickname: 昵称文本
            position: 在文本中的位置
            nickname_analysis: 昵称分析结果
            
        Returns:
            是否为群聊名称
        """
        nickname_info = nickname_analysis['nickname_info']
        first_nickname_index = nickname_analysis['first_nickname_index']
        
        # 检查条件：
        # 1. 是第一个出现的昵称
        # 2. 该昵称只出现一次（或者所有出现都没有内容）
        # 3. 该昵称没有内容
        
        if nickname not in nickname_info:
            return False
        
        info = nickname_info[nickname]
        
        # 条件1: 是第一个出现的昵称
        is_first_nickname = (position == first_nickname_index)
        
        # 条件2: 只出现一次，或者所有出现都没有内容
        is_unique_or_no_content = (info['count'] == 1 or not info['has_content'])
        
        # 条件3: 当前这个昵称没有内容（已经在调用处检查过了）
        
        return is_first_nickname and is_unique_or_no_content
    
    def _is_time_content(self, text: str) -> bool:
        """
        检查文本是否为时间内容
        
        Args:
            text: 要检查的文本
            
        Returns:
            是否为时间内容
        """
        # 时间模式列表（与ContentMarker保持一致）
        time_patterns = [
            r'\d{4}年\d{1,2}月\d{1,2}日\d{1,2}:\d{2}',                          # 2025年6月17日9:10
            r'\d{4}年\d{1,2}月\d{1,2}日',                                       # 2025年6月17日
            r'(昨天|今天|前天|明天)(早上|上午|中午|下午|晚上|凌晨)?\d{1,2}:\d{2}',  # 昨天晚上6:23
            r'(上午|下午|早上|中午|晚上|凌晨)\d{1,2}:\d{2}',                     # 上午9:17
            r'\d{1,2}:\d{2}',                                                  # 5:51
            r'\d{4}-\d{1,2}-\d{1,2}',                                          # 年-月-日  
            r'\d{1,2}/\d{1,2}',                                                # 月/日
            r'\d{1,2}月\d{1,2}日',                                             # 月日
            r'\d{1,2}:\d{2}:\d{2}',                                           # 时:分:秒
            r'(昨天|今天|前天|明天)',                                           # 相对日期
            r'周[一二三四五六日天]',                                            # 星期
        ]
        
        text = text.strip()
        
        # 排除过长的文本（超过30个字符的很可能不是纯时间）
        if len(text) > 30:
            return False
            
        # 排除包含明显非时间关键词的文本
        exclude_keywords = ['报送', '回执', '会议', '参加', '人员', '工作', '通知', '安排', '要求', '地点', '内容', 
                          '完成', '需要', '前', '后', '开始', '结束', '传包', '表格', '填写', '更新', '自测']
        if any(keyword in text for keyword in exclude_keywords):
            return False
        
        # 检查时间模式
        for pattern in time_patterns:
            if re.search(pattern, text):
                match = re.search(pattern, text)
                if match:
                    matched_length = len(match.group())
                    match_ratio = matched_length / len(text)
                    
                    # 对于复合时间格式，降低阈值要求
                    if pattern.startswith('(昨天|今天|前天|明天)') or pattern.startswith('(上午|下午|早上|中午|晚上|凌晨)'):
                        # 复合时间格式，匹配40%以上即可
                        if match_ratio >= 0.4:
                            return True
                    elif pattern.startswith(r'\d{4}年'):
                        # 完整日期格式，匹配70%以上
                        if match_ratio >= 0.7:
                            return True
                    else:
                        # 简单时间格式，仍然要求60%以上
                        if match_ratio >= 0.6:
                            return True
        
        return False