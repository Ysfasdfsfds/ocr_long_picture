#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的文本框排除策略处理器
功能：
1. 从OCR结果JSON文件中加载数据
2. 应用三步排除策略（时间戳排除、右半边位置识别、绿色区域验证）
3. 输出排除结果统计
"""

import json
import re
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import DBSCAN


class TextExclusionProcessor:
    """独立的文本框排除策略处理器"""
    
    def __init__(self):
        """初始化处理器"""
        self.timestamp_pattern = r'\d{1,2}月\d{1,2}日'  # 时间戳匹配模式
        
    def load_from_json(self, json_path: str) -> Tuple[List[Dict], int, int]:
        """从JSON文件加载OCR结果"""
        print("=" * 50)
        print("开始加载OCR结果JSON文件...")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取图片尺寸
        image_shape = data.get('image_shape', [])
        if len(image_shape) >= 2:
            image_h, image_w = image_shape[0], image_shape[1]
        else:
            raise ValueError("JSON文件中缺少图片尺寸信息")
        
        # 转换结果格式
        text_info = []
        results = data.get('results', [])
        
        for item in results:
            box = item.get('box', [])
            text = item.get('text', '')
            center = item.get('center', [])
            
            if not box or not center:
                continue
                
            # 计算文本框特征
            center_x, center_y = center[0], center[1]
            x_coords = [pt[0] for pt in box]
            y_coords = [pt[1] for pt in box]
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            aspect_ratio = width / height if height > 0 else 0
            
            text_info.append({
                'id': item.get('id', -1),
                'box': box,
                'text': text,
                'center_x': center_x,
                'center_y': center_y,
                'width': width,
                'height': height,
                'aspect_ratio': aspect_ratio,
                'text_length': len(text),
                'confidence': item.get('confidence', 0.0)
            })
        
        # 按Y坐标排序
        text_info.sort(key=lambda x: x['center_y'])
        
        print(f"成功加载 {len(text_info)} 个文本框")
        print(f"图片尺寸: 宽度={image_w}, 高度={image_h}")
        print("加载完成")
        print("=" * 50)
        
        return text_info, image_w, image_h
    
    def apply_exclusion_strategy(self, text_info: List[Dict], image_w: int, 
                               original_img_path: Optional[str] = None) -> Tuple:
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
        
        # 第三步：通过绿色区域检测验证第二步结果（如果有原图）
        if original_img_path and self._check_image_exists(original_img_path):
            original_img = cv2.imread(original_img_path)
            truly_excluded, false_positives = self._verify_with_green_detection(
                right_position_candidates, original_img
            )
        else:
            print("第三步: 未提供原图路径或图片不存在，跳过绿色区域验证")
            # 不做绿色验证，假设所有右半边候选都是真正需要排除的
            truly_excluded = right_position_candidates
            false_positives = []
        
        # 将误判的文本框重新加回保留列表
        final_remaining = remaining_after_step2 + false_positives
        
        # 打印排除统计
        self._print_exclusion_summary(
            text_info, excluded_timestamps, truly_excluded, 
            false_positives, final_remaining
        )
        
        return final_remaining, excluded_timestamps, truly_excluded, false_positives
    
    def _exclude_timestamps(self, text_info: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
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
    
    def _identify_right_position_candidates(self, remaining_texts: List[Dict], 
                                          half_width: float) -> Tuple[List[Dict], List[Dict]]:
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
    
    def _verify_with_green_detection(self, candidates: List[Dict], 
                                   original_img: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
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
    
    def _is_green_area(self, text_info: Dict, original_img: np.ndarray) -> bool:
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
    
    def _check_image_exists(self, img_path: str) -> bool:
        """检查图片文件是否存在"""
        try:
            import os
            return os.path.exists(img_path)
        except:
            return False
    
    def _print_exclusion_summary(self, original_texts: List[Dict], excluded_timestamps: List[Dict], 
                                truly_excluded: List[Dict], false_positives: List[Dict], 
                                final_remaining: List[Dict]):
        """打印排除策略汇总"""
        total_excluded = len(excluded_timestamps) + len(truly_excluded)
        
        print(f"\n排除策略汇总:")
        print(f"  原始文本框: {len(original_texts)}个")
        print(f"  第一步识别时间戳组: {len(excluded_timestamps)}个")
        print(f"  第二步识别右半边候选: {len(truly_excluded) + len(false_positives)}个")
        print(f"  第三步确认右边组: {len(truly_excluded)}个")
        print(f"  第三步恢复误判: {len(false_positives)}个")
        
        print("\n=== 第一步排除的时间戳组 ===")
        if excluded_timestamps:
            for i, info in enumerate(excluded_timestamps):
                print(f"{i+1}. {info['text']} (center_x: {info['center_x']:.1f}, center_y: {info['center_y']:.1f})")
        else:
            print("无识别到的时间戳。")
        
        print("\n=== 第三步识别的右边组 ===")
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
        
        # 打印最终保留的文本框列表
        print(f"\n=== 最终保留的文本框 ===")
        for i, info in enumerate(final_remaining):
            print(f"{i+1}. {info['text']} (center_x: {info['center_x']:.1f}, center_y: {info['center_y']:.1f})")
    
    def save_results(self, output_path: str, final_remaining: List[Dict], 
                    excluded_timestamps: List[Dict], truly_excluded: List[Dict], 
                    false_positives: List[Dict]):
        """保存排除结果到JSON文件"""
        result = {
            "final_remaining": final_remaining,
            "excluded_timestamps": excluded_timestamps,
            "truly_excluded": truly_excluded,
            "false_positives": false_positives,
            "summary": {
                "total_remaining": len(final_remaining),
                "total_excluded_timestamps": len(excluded_timestamps),
                "total_excluded_right": len(truly_excluded),
                "total_recovered": len(false_positives)
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存到: {output_path}")
    
    def perform_clustering(self, left_half_info: List[Dict]) -> Tuple[List[int], int]:
        """第四步：对左半边文本框进行DBSCAN聚类"""
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
        
        # 对长图片调整参数
        if y_range > 1000:  # 判断是否为长图
            # 大幅降低eps值
            eps = min(y_std * 0.05, y_range * 0.002, 20.0)  # 使用非常小的聚类距离
            print(f"检测到长图片，使用非常小的聚类距离: eps={eps:.1f}")
            
            # 打印Y坐标分布特征，用于调试
            y_diffs = [left_y_coords[i+1] - left_y_coords[i] for i in range(len(left_y_coords)-1)]
            avg_diff = sum(y_diffs) / len(y_diffs) if y_diffs else 0
            median_diff = sorted(y_diffs)[len(y_diffs)//2] if y_diffs else 0
            print(f"Y坐标平均间距: {avg_diff:.1f}, 中位数间距: {median_diff:.1f}")
            print(f"最小间距: {min(y_diffs):.1f}, 最大间距: {max(y_diffs):.1f}")
            
            # 根据Y坐标间距自适应调整eps
            eps = max(median_diff * 2.0, 20.0)
            print(f"基于Y坐标间距调整后的eps值: {eps:.1f}")
        else:
            eps = max(y_std * 0.15, y_range * 0.02, 100.0)  # 普通图片使用原参数
            
        min_samples = 1  # 保持最小样本数为1
        
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
    
    def organize_message_groups(self, left_half_info: List[Dict], labels: List[int]) -> Tuple[Dict, List]:
        """第五步：组织消息组数据"""
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
    
    def print_clustering_results(self, sorted_groups: Dict, ordered_group_ids: List):
        """第六步：打印聚类分析结果"""
        print(f"\n=== 消息组分析结果（从上到下） ===")
        for i, group_id in enumerate(ordered_group_ids):
            group_info = sorted_groups[group_id]
            group_name = f"噪声组" if group_id == -1 else f"消息组{i+1}"
            print(f"{group_name} (聚类ID={group_id}): {group_info['count']}个文本框")
            print(f"  Y坐标范围: {group_info['min_y']:.1f} - {group_info['max_y']:.1f}")
            
            for j, text_info in enumerate(group_info['texts']):
                print(f"    {j+1}. {text_info['text']} (y={text_info['center_y']:.1f})")
            print()
    
    def create_final_groups(self, sorted_groups: Dict, ordered_group_ids: List, 
                          excluded_timestamps: List[Dict], truly_excluded: List[Dict], 
                          false_positives: List[Dict]) -> List[Dict]:
        """第七步：创建最终的文本组结构"""
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
    
    def print_detailed_results(self, all_text_groups: List[Dict]) -> List[Dict]:
        """第八步：打印详细的分组结果"""
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


def main():
    """主函数"""
    # 初始化处理器
    processor = TextExclusionProcessor()
    
    # JSON文件路径
    json_path = "output_json/long_picture_ocr_result.json"
    
    # 原图路径（可选，用于绿色区域验证）
    original_img_path = "images/long_picture.jpg"  # 如果没有可设为None
    
    try:
        # 第一步：从JSON加载数据
        text_info, image_w, image_h = processor.load_from_json(json_path)
        
        # 第二步：应用排除策略
        final_remaining, excluded_timestamps, truly_excluded, false_positives = \
            processor.apply_exclusion_strategy(text_info, image_w, original_img_path)
        
        # 第三步：保存排除结果
        output_path = "output_json/exclusion_result.json"
        processor.save_results(output_path, final_remaining, excluded_timestamps, 
                             truly_excluded, false_positives)
        
        # 第四步：DBSCAN聚类
        labels, n_clusters = processor.perform_clustering(final_remaining)
        
        # 第五步：组织消息组
        sorted_groups, ordered_group_ids = processor.organize_message_groups(final_remaining, labels)
        
        # 第六步：打印聚类结果
        processor.print_clustering_results(sorted_groups, ordered_group_ids)
        
        # 第七步：创建最终组结构
        all_text_groups = processor.create_final_groups(
            sorted_groups, ordered_group_ids, 
            excluded_timestamps, truly_excluded, false_positives
        )
        
        # 第八步：打印详细结果
        message_conversations = processor.print_detailed_results(all_text_groups)
        
        # 保存最终结果
        final_output_path = "output_json/final_grouped_result.json"
        final_result = {
            "all_text_groups": all_text_groups,
            "message_conversations": message_conversations,
            "clustering_info": {
                "n_clusters": n_clusters,
                "labels": labels.tolist() if hasattr(labels, 'tolist') else labels
            },
            "summary": {
                "total_groups": len(all_text_groups),
                "total_conversations": len(message_conversations)
            }
        }
        
        with open(final_output_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        
        print(f"\n最终结果已保存到: {final_output_path}")
        print(f"\n处理完成！")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
