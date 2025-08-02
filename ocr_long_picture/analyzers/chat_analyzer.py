#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聊天分析模块：负责聊天消息结构化
"""

import re
from typing import List, Dict

from ..config.settings import TIME_PATTERNS, EXCLUDE_KEYWORDS


class ChatAnalyzer:
    """聊天分析模块：负责聊天消息结构化"""
    
    def organize_chat_messages(self, marked_texts: List[str]) -> List[Dict]:
        """将标记后的文本组织成结构化聊天消息"""
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
                        break
                
                if my_content_parts:
                    combined_content = " ".join(my_content_parts)
                    messages.append({
                        "type": "my_chat",
                        "昵称": "我",
                        "内容": combined_content
                    })
                
                i = j
                continue
            
            # 处理昵称标记
            if "(昵称)" in text:
                nickname = text.replace("(昵称)", "").strip()
                
                content_parts = []
                retract_messages = []
                j = i + 1
                
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
                            if "撤回了一条消息" in content:
                                retract_messages.append(content)
                                print(f"检测到撤回消息: {content}")
                            elif self._is_time_content(content):
                                messages.append({
                                    "type": "time",
                                    "time": content
                                })
                                print(f"在内容中检测到时间: {content}")
                            else:
                                content_parts.append(content)
                    else:
                        content_parts.append(next_text)
                    
                    j += 1
                
                if content_parts:
                    combined_content = " ".join(content_parts)
                    messages.append({
                        "type": "chat",
                        "昵称": nickname,
                        "内容": combined_content
                    })
                else:
                    if self._is_group_name(nickname, i, nickname_analysis):
                        messages.append({
                            "type": "group_name",
                            "群聊名称": nickname
                        })
                        print(f"检测到群聊名称: {nickname}")
                
                for retract_content in retract_messages:
                    messages.append({
                        "type": "retract_message",
                        "撤回信息": retract_content
                    })
                
                i = j
                continue
            
            # 处理孤立的内容标记
            if "(内容)" in text:
                content = text.replace("(内容)", "").strip()
                if "撤回了一条消息" in content:
                    messages.append({
                        "type": "retract_message",
                        "撤回信息": content
                    })
                    print(f"检测到撤回消息: {content}")
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
        """分析所有昵称的出现情况"""
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
        """判断是否为群聊名称"""
        nickname_info = nickname_analysis['nickname_info']
        first_nickname_index = nickname_analysis['first_nickname_index']
        
        if nickname not in nickname_info:
            return False
        
        info = nickname_info[nickname]
        
        is_first_nickname = (position == first_nickname_index)
        is_unique_or_no_content = (info['count'] == 1 or not info['has_content'])
        
        return is_first_nickname and is_unique_or_no_content
    
    def _is_time_content(self, text: str) -> bool:
        """检查文本是否为时间内容"""
        text = text.strip()
        
        if len(text) > 30:
            return False
            
        if any(keyword in text for keyword in EXCLUDE_KEYWORDS):
            return False
        
        for pattern in TIME_PATTERNS:
            if re.search(pattern, text):
                match = re.search(pattern, text)
                if match:
                    matched_length = len(match.group())
                    match_ratio = matched_length / len(text)
                    
                    if pattern.startswith('(昨天|今天|前天|明天)') or pattern.startswith('(上午|下午|早上|中午|晚上|凌晨)'):
                        if match_ratio >= 0.4:
                            return True
                    elif pattern.startswith(r'\d{4}年'):
                        if match_ratio >= 0.7:
                            return True
                    else:
                        if match_ratio >= 0.6:
                            return True
        
        return False