#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
微信聊天截图OCR识别与消息分组聚类系统
功能：
1. OCR识别微信聊天截图中的文本框
2. 智能排除时间戳和右半边文本框
3. 基于DBSCAN算法对左半边消息进行聚类分组
4. 输出结构化的聊天消息组
"""

import cv2
import re
import numpy as np
import requests
from sklearn.cluster import DBSCAN
from rapidocr import RapidOCR


class WeChatOCRProcessor:
    """微信聊天截图OCR处理器"""
    
    def __init__(self, config_path="default_rapidocr.yaml"):
        """初始化OCR引擎"""
        self.engine = RapidOCR(config_path=config_path)
        self.timestamp_pattern = r'\d{1,2}月\d{1,2}日'  # 时间戳匹配模式
        
    def load_and_recognize(self, img_path):
        """加载图片并进行OCR识别"""
        print("=" * 50)
        print("开始OCR识别...")
        
        # 获取原图尺寸
        original_img = cv2.imread(img_path)
        if original_img is None:
            raise ValueError(f"无法读取图片: {img_path}")
        
        image_h, image_w = original_img.shape[:2]
        print(f"原图尺寸: 宽度={image_w}, 高度={image_h}")
        
        # OCR识别
        result = self.engine(img_path)
        
        # 检查识别结果
        if result.boxes is None or len(result.boxes) == 0:  # type: ignore
            raise ValueError("没有识别到任何文本框")
        
        print(f"总共识别到 {len(result.boxes)} 个文本框")  # type: ignore
        print("识别完成")
        print("=" * 50)
        
        return result, image_w, image_h, original_img
    
    def extract_text_info(self, result):
        """提取文本框详细信息"""
        text_info = []
        
        # 计算每个文本框的中心坐标和其他特征
        for i, (box, text) in enumerate(zip(result.boxes, result.txts)):  # type: ignore
            center_x = np.mean([pt[0] for pt in box])
            center_y = np.mean([pt[1] for pt in box])
            width = max([pt[0] for pt in box]) - min([pt[0] for pt in box])
            height = max([pt[1] for pt in box]) - min([pt[1] for pt in box])
            aspect_ratio = width / height if height > 0 else 0
            
            text_info.append({
                'box': box,
                'text': text,
                'center_x': center_x,
                'center_y': center_y,
                'width': width,
                'height': height,
                'aspect_ratio': aspect_ratio,
                'text_length': len(text)
            })
        
        # 按Y坐标排序
        text_info.sort(key=lambda x: x['center_y'])
        print(f"分析 {len(text_info)} 个文本框的布局特征")
        
        return text_info
    
    def apply_exclusion_strategy(self, text_info, image_w, original_img):
        """应用三步排除策略"""
        print("\n" + "=" * 30)
        print("开始三步排除策略")
        print("=" * 30)
        
        half_width = image_w / 2
        print(f"图片宽度的一半: {half_width}")
        
        all_x = [info['center_x'] for info in text_info]
        print(f"文本框X坐标范围: {min(all_x):.1f} - {max(all_x):.1f}")
        
        # 第一步：排除时间戳
        excluded_timestamps, remaining_after_step1 = self._exclude_timestamps(text_info)
        
        # 第二步：识别右半边位置候选（可能有误判）
        right_position_candidates, remaining_after_step2 = self._identify_right_position_candidates(
            remaining_after_step1, half_width
        )
        
        # 第三步：通过绿色区域检测验证第二步结果
        truly_excluded, false_positives = self._verify_with_green_detection(
            right_position_candidates, original_img
        )
        
        # 将误判的文本框重新加回保留列表
        final_remaining = remaining_after_step2 + false_positives
        
        # 打印排除统计
        self._print_exclusion_summary(
            text_info, excluded_timestamps, truly_excluded, 
            false_positives, final_remaining
        )
        
        return final_remaining, excluded_timestamps, truly_excluded, false_positives
    
    def _exclude_timestamps(self, text_info):
        """第一步：排除时间戳"""
        excluded_timestamps = []
        remaining_after_step1 = []
        
        for info in text_info:
            if re.search(self.timestamp_pattern, info['text']):
                excluded_timestamps.append(info)
                print(f"第一步排除时间戳: {info['text']} (center_x: {info['center_x']:.1f})")
            else:
                remaining_after_step1.append(info)
        
        print(f"第一步完成: 排除时间戳 {len(excluded_timestamps)}个，剩余 {len(remaining_after_step1)}个")
        return excluded_timestamps, remaining_after_step1
    
    def _exclude_right_position(self, remaining_texts, half_width):
        """第二步：排除右半边位置"""
        excluded_right_position = []
        left_half_info = []
        
        for info in remaining_texts:
            if info['center_x'] > half_width:
                excluded_right_position.append(info)
                print(f"第二步排除右半边: {info['text']} (center_x: {info['center_x']:.1f})")
            else:
                left_half_info.append(info)
        
        print(f"第二步完成: 排除右半边位置 {len(excluded_right_position)}个，剩余 {len(left_half_info)}个")
        return excluded_right_position, left_half_info
    
    def _identify_right_position_candidates(self, remaining_texts, half_width):
        """第二步：识别右半边位置候选（可能有误判）"""
        right_position_candidates = []
        left_half_info = []
        
        for info in remaining_texts:
            if info['center_x'] > half_width:
                right_position_candidates.append(info)
                print(f"第二步识别右半边候选: {info['text']} (center_x: {info['center_x']:.1f})")
            else:
                left_half_info.append(info)
        
        print(f"第二步完成: 识别右半边候选 {len(right_position_candidates)}个，剩余 {len(left_half_info)}个")
        return right_position_candidates, left_half_info
    
    def _verify_with_green_detection(self, candidates, original_img):
        """第三步：通过绿色区域检测验证候选文本框"""
        truly_excluded = []  # 确实是绿色区域，真正排除
        false_positives = []  # 不是绿色区域，误判需要恢复
        
        print(f"第三步开始: 对{len(candidates)}个候选进行绿色区域验证")
        
        for info in candidates:
            if self._is_green_area(info, original_img):
                truly_excluded.append(info)
                print(f"第三步确认排除: {info['text']} (确实是绿色区域)")
            else:
                false_positives.append(info)
                print(f"第三步恢复保留: {info['text']} (不是绿色区域，误判)")
        
        print(f"第三步完成: 确认排除 {len(truly_excluded)}个，恢复保留 {len(false_positives)}个")
        return truly_excluded, false_positives
    

    
    def _is_green_area(self, text_info, original_img):
        """检测文本框区域是否为绿色"""
        try:
            # 获取文本框的边界坐标
            box = text_info['box']
            x_coords = [pt[0] for pt in box]
            y_coords = [pt[1] for pt in box]
            
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            # 确保坐标在图片范围内
            h, w = original_img.shape[:2]
            x_min = max(0, x_min)
            x_max = min(w, x_max)
            y_min = max(0, y_min)
            y_max = min(h, y_max)
            
            # 裁剪文本框区域
            if x_max <= x_min or y_max <= y_min:
                return False
                
            crop_region = original_img[y_min:y_max, x_min:x_max]
            
            # 转换为HSV颜色空间
            hsv = cv2.cvtColor(crop_region, cv2.COLOR_BGR2HSV)
            
            # 定义绿色的HSV范围
            # 绿色在HSV中的范围：H(35-85), S(40-255), V(40-255)
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            
            # 创建绿色掩码
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # 计算绿色像素比例
            total_pixels = crop_region.shape[0] * crop_region.shape[1]
            green_pixels = np.sum(green_mask > 0)
            green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0
            
            # 如果绿色像素超过30%，认为是绿色区域
            is_green = green_ratio > 0.3
            
            if is_green:
                print(f"    检测到绿色区域: {text_info['text']} (绿色比例: {green_ratio:.2%})")
            
            return is_green
            
        except Exception as e:
            print(f"    绿色检测失败: {text_info['text']} (错误: {e})")
            return False
    
    def _print_exclusion_summary(self, original_texts, excluded_timestamps, 
                                truly_excluded, false_positives, final_remaining):
        """打印排除策略汇总"""
        total_excluded = len(excluded_timestamps) + len(truly_excluded)
        
        print(f"\n排除策略汇总:")
        print(f"  原始文本框: {len(original_texts)}个")
        print(f"  第一步识别时间戳组: {len(excluded_timestamps)}个")
        print(f"  第二步识别右半边候选: {len(truly_excluded) + len(false_positives)}个")
        print(f"  第三步确认右边组: {len(truly_excluded)}个")
        print(f"  第三步恢复误判: {len(false_positives)}个")
        
        print("\n=== 第三步识别的右边组（绿色区域） ===")
        if truly_excluded:
            for i, info in enumerate(truly_excluded):
                print(f"{i+1}. {info['text']} (center_x: {info['center_x']:.1f}, center_y: {info['center_y']:.1f})")
        else:
            print("无识别到的右边组。")
        
        print("\n=== 第三步恢复的误判文本框 ===")
        if false_positives:
            for i, info in enumerate(false_positives):
                print(f"{i+1}. {info['text']} (center_x: {info['center_x']:.1f}, center_y: {info['center_y']:.1f}) - 已恢复到保留列表")
        else:
            print("无恢复的误判文本框。")
        
        print(f"\n  总共排除: {total_excluded}个")
        print(f"  最终保留: {len(final_remaining)}个")
    
    def perform_clustering(self, left_half_info):
        """对左半边文本框进行DBSCAN聚类"""
        if len(left_half_info) == 0:
            raise ValueError("左半边没有文本框，无法进行聚类")
        
        print("\n" + "=" * 30)
        print("开始DBSCAN聚类分析")
        print("=" * 30)
        
        # 提取Y坐标用于聚类
        left_y_coords = [info['center_y'] for info in left_half_info]
        y_coords_array = np.array(left_y_coords).reshape(-1, 1)
        
        print(f"左半边Y坐标分布: {sorted(left_y_coords)}")
        
        # 计算DBSCAN参数
        y_std = float(np.std(left_y_coords))
        y_range = float(max(left_y_coords) - min(left_y_coords))
        eps = max(y_std * 0.15, y_range * 0.02, 100.0)  # 精确控制聚类距离
        min_samples = 1
        
        print(f"消息组DBSCAN参数: eps={eps:.1f}, min_samples={min_samples}")
        
        # 执行聚类
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(y_coords_array)
        
        # 分析聚类结果
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"消息组聚类结果: {n_clusters}个消息组, {n_noise}个噪声点")
        print(f"聚类标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        return labels, n_clusters
    
    def organize_message_groups(self, left_half_info, labels):
        """组织消息组数据"""
        # 按聚类分组
        message_groups = {}
        for i, info in enumerate(left_half_info):
            group_id = labels[i]
            if group_id not in message_groups:
                message_groups[group_id] = []
            message_groups[group_id].append(info)
        
        # 按Y坐标排序每个组内的文本框
        sorted_groups = {}
        for group_id, group_texts in message_groups.items():
            group_texts.sort(key=lambda x: x['center_y'])
            sorted_groups[group_id] = {
                'texts': group_texts,
                'min_y': min(text['center_y'] for text in group_texts),
                'max_y': max(text['center_y'] for text in group_texts),
                'count': len(group_texts)
            }
        
        # 按组的Y坐标从上到下排序
        ordered_group_ids = sorted(sorted_groups.keys(), 
                                 key=lambda gid: sorted_groups[gid]['min_y'])
        
        return sorted_groups, ordered_group_ids
    
    def print_clustering_results(self, sorted_groups, ordered_group_ids):
        """打印聚类分析结果"""
        print(f"\n=== 消息组分析结果（从上到下） ===")
        for i, group_id in enumerate(ordered_group_ids):
            group_info = sorted_groups[group_id]
            group_name = f"噪声组" if group_id == -1 else f"消息组{i+1}"
            print(f"{group_name} (聚类ID={group_id}): {group_info['count']}个文本框")
            print(f"  Y坐标范围: {group_info['min_y']:.1f} - {group_info['max_y']:.1f}")
            
            for j, text_info in enumerate(group_info['texts']):
                print(f"    {j+1}. {text_info['text']} (y={text_info['center_y']:.1f})")
            print()
    
    def create_final_groups(self, sorted_groups, ordered_group_ids, 
                          excluded_timestamps, truly_excluded, false_positives):
        """创建最终的文本组结构"""
        all_text_groups = []
        
        # 添加消息组
        for group_id in ordered_group_ids:
            group_info = sorted_groups[group_id]
            group_name = f"噪声组" if group_id == -1 else f"消息组{len(all_text_groups)+1}"
            
            message_group = {
                'name': group_name,
                'texts': group_info['texts'],
                'type': 'message_group'
            }
            all_text_groups.append(message_group)
        
        # 添加时间戳组
        if excluded_timestamps:
            excluded_timestamps.sort(key=lambda x: x['center_y'])
            timestamp_group = {
                'name': '时间戳组',
                'texts': excluded_timestamps,
                'type': 'timestamp_group'
            }
            all_text_groups.append(timestamp_group)
        
        # 添加右边组（绿色区域）
        if truly_excluded:
            truly_excluded.sort(key=lambda x: x['center_y'])
            green_group = {
                'name': '右边组',
                'texts': truly_excluded,
                'type': 'right_side_group'
            }
            all_text_groups.append(green_group)
        
        # 添加恢复的误判文本框组（仅用于展示，实际已恢复到保留列表）
        if false_positives:
            false_positives.sort(key=lambda x: x['center_y'])
            recovered_group = {
                'name': '已恢复的误判文本框',
                'texts': false_positives,
                'type': 'recovered_group'
            }
            all_text_groups.append(recovered_group)
        
        return all_text_groups
    
    def print_detailed_results(self, all_text_groups):
        """打印详细的分组结果"""
        print(f"\n" + "=" * 50)
        print("消息组分类详细结果")
        print("=" * 50)
        
        # 打印每组的简要信息
        for group in all_text_groups:
            print(f"\n{group['name']} ({group['type']}):")
            for i, text_info in enumerate(group['texts']):
                box_area = text_info['width'] * text_info['height']
                print(f"  {i+1}. {text_info['text']} "
                      f"(x={text_info['center_x']:.1f}, y={text_info['center_y']:.1f}, "
                      f"面积={box_area:.0f})")
        
        # 打印每组的完整对话
        print(f"\n=== 每组消息的完整对话 ===")
        message_conversations = []
        for group in all_text_groups:
            if group['type'] == 'message_group':
                conversation = " ".join([text_info['text'] for text_info in group['texts']])
                message_conversations.append({
                    'group_name': group['name'],
                    'conversation': conversation,
                    'text_count': len(group['texts'])
                })
                print(f"{group['name']} ({len(group['texts'])}个文本框): {conversation}")
        
        # 打印分组统计
        print(f"\n=== 分组统计 ===")
        for group in all_text_groups:
            print(f"{group['name']}数量: {len(group['texts'])}")
        
        return message_conversations
    
    def send_to_llm(self, all_text_groups):
        """发送结果到LLM进行翻译"""
        # 构建用于LLM的JSON结构内容
        import json
        
        # 收集所有文本项，按y坐标排序后重新组织
        all_items = []
        
        for group in all_text_groups:
            # 排除已恢复的误判文本框组（它们已经恢复到保留列表中了）
            if group['type'] == 'recovered_group':
                continue
            
            group_name = group['name']
            group_type = group['type']
            
            # 将每个组的文本项添加到总列表中
            for text_info in group['texts']:
                all_items.append({
                    'text_info': text_info,
                    'group_name': group_name,
                    'group_type': group_type,
                    'center_y': text_info['center_y']
                })
        
        # 按Y坐标排序所有文本项
        all_items.sort(key=lambda x: x['center_y'])
        
        # 重新构建JSON，按y坐标顺序混合时间戳和消息组
        result_json = {}
        current_section = 1
        
        i = 0
        while i < len(all_items):
            item = all_items[i]
            
            if item['group_type'] == 'timestamp_group':
                # 时间戳单独作为一个条目
                section_name = f"时间戳{current_section}"
                result_json[section_name] = [{"text": item['text_info']['text']}]
                current_section += 1
                i += 1
            else:
                # 收集同一组的所有文本
                section_name = f"对话{current_section}"
                section_items = []
                current_group = item['group_name']
                group_texts = []
                
                # 收集当前组的所有文本
                while i < len(all_items) and all_items[i]['group_name'] == current_group:
                    group_texts.append(all_items[i]['text_info'])
                    i += 1
                
                # 右边组特殊处理：不使用center_x最小逻辑
                if item['group_type'] == 'right_side_group':
                    # 右边组的所有文本都标记为text
                    for text_info in group_texts:
                        section_items.append({"text": text_info['text']})
                    # 添加固定的"person":"我"
                    section_items.append({"person": "我"})
                else:
                    # 其他组：找到center_x最小的文本（发言人）
                    if group_texts:
                        min_x_index = min(range(len(group_texts)), key=lambda idx: group_texts[idx]['center_x'])
                    
                    for idx, text_info in enumerate(group_texts):
                        if idx == min_x_index:
                            section_items.append({"person": text_info['text']})
                        else:
                            section_items.append({"text": text_info['text']})
                
                result_json[section_name] = section_items
                current_section += 1
        
        # 将JSON转换为字符串
        result_to_llms = json.dumps(result_json, ensure_ascii=False, indent=2)
        
        print(f"\n=== 组合后的JSON数据（用于LLM） ===")
        print(result_to_llms)
        
        # 获取用户问题
        print("\n" + "="*50)
        print("微信聊天解读助手已准备就绪！")
        print("您可以询问关于这个聊天记录的任何问题，例如：")
        print("  - 他们在讨论什么？")
        print("  - 找到那个设备了吗？")
        print("  - 谁回复了什么？")
        print("  - 这个聊天的主要内容是什么？")
        print("-" * 50)
        print("请输入您的问题（直接按回车使用默认问题）：")
        user_question = input(">>> ")
        
        if not user_question.strip():
            user_question = "请帮我分析这个微信聊天记录的主要内容和关键信息"
            print(f"使用默认问题: {user_question}")
        
        # 调用本地Ollama API
        ollama_url = "http://localhost:11434/api/generate"
        model_name = "qwen3:8b"
        
        payload = {
            "model": model_name,
            "prompt": f"用户问题：{user_question}",
            "system": f"你是微信聊天解读助手，需要根据提供的JSON格式微信聊天上下文回答用户的问题。JSON数据包含了不同的消息组、时间戳组和右边组，每个组都有相应的文本内容和坐标信息。上下文JSON数据：{result_to_llms}",
            "stream": False
        }
        
        print("正在将识别结果发送给本地Ollama的qwen3:8b模型...")
        
        try:
            response = requests.post(ollama_url, json=payload)
            response.raise_for_status()
            data = response.json()
            llm_output = data.get("response", "")
            print("Ollama模型返回结果：")
            print(llm_output)
        except Exception as e:
            print("调用Ollama接口时出错：", e)


def main():
    """主函数"""
    # 初始化处理器
    processor = WeChatOCRProcessor()
    
    # 图片路径
    img_path = "images/111.jpeg"
    
    try:
        # 第一步：OCR识别
        result, image_w, image_h, original_img = processor.load_and_recognize(img_path)
        
        # 第二步：提取文本信息
        text_info = processor.extract_text_info(result)
        
        # 第三步：应用排除策略
        left_half_info, excluded_timestamps, truly_excluded, false_positives = \
            processor.apply_exclusion_strategy(text_info, image_w, original_img)
        
        # 第四步：DBSCAN聚类
        labels, n_clusters = processor.perform_clustering(left_half_info)
        
        # 第五步：组织消息组
        sorted_groups, ordered_group_ids = processor.organize_message_groups(left_half_info, labels)
        
        # 第六步：打印聚类结果
        processor.print_clustering_results(sorted_groups, ordered_group_ids)
        
        # 第七步：创建最终组结构
        all_text_groups = processor.create_final_groups(
            sorted_groups, ordered_group_ids, 
            excluded_timestamps, truly_excluded, false_positives
        )
        
        # 第八步：打印详细结果
        message_conversations = processor.print_detailed_results(all_text_groups)
        
        # 第九步：发送到LLM
        processor.send_to_llm(all_text_groups)
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
