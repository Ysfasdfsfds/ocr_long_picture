#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
长图OCR处理脚本
实现长图切分、OCR识别、结果整合和可视化
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
from rapidocr import RapidOCR
import math
import shutil
from LLM_run import process_with_llm
import re
from sklearn.cluster import DBSCAN
from process_avatar import process_avatar,process_avatar_v2

class LongImageOCR:
    def __init__(self, config_path: str = "default_rapidocr.yaml"):
        """
        初始化长图OCR处理器
        
        Args:
            config_path: RapidOCR配置文件路径
        """
        self.engine = RapidOCR(config_path=config_path)
        self.slice_height = 1200  # 切片高度
        self.overlap = 200  # 重叠区域像素
        self.text_score_threshold = 0.65  # 文本识别置信度阈值
        self.original_image = None  # 存储原始图像
        
        # 汇总字段：存储原图坐标系统中的所有结果
        self.all_ocr_results_original = []  # 所有OCR结果(原图坐标)
        self.all_avatar_positions_original = []  # 所有头像位置(原图坐标)
        self.marked_ocr_results_original = []  # 标记后的OCR结果(原图坐标)
        self.structured_chat_messages = []  # 结构化的聊天消息
        
        # 创建输出目录
        self.output_json_dir = Path("./output_json")
        self.output_images_dir = Path("./output_images")
        self.output_json_dir.mkdir(exist_ok=True)
        self.output_images_dir.mkdir(exist_ok=True)
        
    def slice_image(self, image_path: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        切分长图
        
        Args:
            image_path: 图像路径
            
        Returns:
            original_image: 原始图像
            slices_info: 切片信息列表，包含切片图像和位置信息
        """
        # 读取原始图像
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"无法读取图像: {image_path}")
            
        h, w, c = original_image.shape
        print(f"原始图像尺寸: {w} x {h}")
        
        if h <= self.slice_height:
            # 图像高度小于等于切片高度，不需要切分
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
            # 计算当前切片的结束位置
            end_y = min(current_y + self.slice_height, h)
            
            # 提取切片
            slice_img = original_image[current_y:end_y, :, :]
            
            # 保存切片信息
            slice_info = {
                'slice': slice_img,
                'start_y': current_y,
                'end_y': end_y,
                'slice_index': slice_index
            }
            slices_info.append(slice_info)
            
            # 保存切片图像
            slice_path = self.output_images_dir / f"slice_{slice_index:03d}.jpg"
            cv2.imwrite(str(slice_path), slice_img)
            print(f"保存切片 {slice_index}: {slice_path}")
            
            # 计算下一个切片的起始位置
            if end_y >= h:
                break
            current_y = end_y - self.overlap
            slice_index += 1
            
        return original_image, slices_info
    
    def process_slices(self, slices_info: List[Dict]) -> List[Dict]:
        """
        对所有切片进行OCR处理和聊天消息分析
        
        Args:
            slices_info: 切片信息列表
            
        Returns:
            slice_results: 每个切片的OCR和聊天分析结果列表
        """
        slice_results = []
        
        # 初始化汇总字段：存储原图坐标系统中的所有OCR结果和头像位置
        all_ocr_results_original = []  # 所有OCR结果(原图坐标)
        all_avatar_positions_original = []  # 所有头像位置(原图坐标)

        #-------------------------------------------------------------
        #基于所有切片，计算x_croped
        from process_avatar import preprocess_and_crop_image, slice_x_croped_values
        # 根据切片数量决定处理逻辑：
        # - 如果只有一个切片：包括所有
        # 根据切片数量确定处理策略
        total_slices = len(slices_info)
        if total_slices == 1:
            # 只有一个切片，处理所有切片
            slices_to_process = slices_info
            print("只有一个切片，将处理所有切片")
        elif total_slices == 2:
            # 两个切片，选择第一个切片
            slices_to_process = slices_info[:1]
            print("有2个切片，将只处理第一个切片")
        else:
            # 大于等于3个切片，排除开始和结束的切片，只处理中间切片
            slices_to_process = slices_info[1:-1]
            print(f"共有{total_slices}个切片，将处理中间{len(slices_to_process)}个切片（排除第一个和最后一个）")
        
        for index,slice_info in enumerate(slices_to_process):
            img, binary, rects = preprocess_and_crop_image(slice_info['slice'], index, slice_info['start_y'])
        
        # 处理slice_x_croped_values中的所有target_box
        print("开始处理slice_x_croped_values中的target_box...")
        
        #通过一些策略，从众多target_box中，找到合适框，作为x_croped的值
        # 收集所有target_box并按x坐标排序
        all_boxes = []
        for slice_idx, target_box in slice_x_croped_values.items():
            if target_box is not None:
                # target_box是单个tuple (x, y, w, h)，不是list
                if isinstance(target_box, (list, tuple)) and len(target_box) == 4:
                    x, y, w, h = target_box
                    all_boxes.append((x, y, w, h, slice_idx))
        
        print(f"总共找到 {len(all_boxes)} 个target_box")
        
        # 按x坐标排序
        all_boxes.sort(key=lambda box: box[0])
        
        if not all_boxes:
            print("未找到任何target_box")
        else:
            # 2. 对最左侧前20%的box进行操作
            left_20_percent_count = max(1, int(len(all_boxes) * 0.2))
            left_boxes = all_boxes[:left_20_percent_count]
            print(f"最左侧前20%的box数量: {left_20_percent_count}")
            
            # 找到符合要求的框
            selected_box = None
            for i, (x, y, w, h, slice_idx) in enumerate(left_boxes):
                # 判断是否严格趋近于正方形（宽高比在0.8-1.2之间）
                aspect_ratio = w / h if h > 0 else 0
                is_square_like = 0.8 <= aspect_ratio <= 1.2
                
                print(f"第{i+1}个左侧框: x={x}, y={y}, w={w}, h={h}, 宽高比={aspect_ratio:.2f}, 是否趋近正方形={is_square_like}")
                
                if is_square_like:
                    selected_box = (x, y, w, h, slice_idx)
                    print(f"找到符合要求的框: 第{i+1}个左侧框，位于slice {slice_idx}")
                    break
            
            if selected_box:
                # 基于选中的框计算x_croped
                x, y, w, h, slice_idx = selected_box
                x_croped = x + w  # 使用框的右边界作为裁剪位置
                print(f"基于选中框计算的x_croped值: {x_croped}")
                
                # 创建单独的字典存储x_croped值，不覆盖原始target_box数据
                slice_x_croped_final = {}
                for slice_idx in slice_x_croped_values.keys():
                    slice_x_croped_final[slice_idx] = x_croped
                print("已计算所有slice的x_croped值")
            else:
                print("未找到符合要求的框（最左侧前20%中没有趋近正方形的框）")
        # 将selected_box画到原图中并保存
        if selected_box:
            x, y, w, h, slice_idx = selected_box
            # 获取对应切片的图像
            for slice_info in slices_info:
                if slice_info['slice_index'] == slice_idx:
                    slice_img = slice_info['slice']
                    # 在切片图像上绘制矩形
                    slice_with_box = slice_img.copy()
                    cv2.rectangle(slice_with_box, (x, y - slice_info['start_y']), 
                                 (x + w, y + h - slice_info['start_y']), (0, 0, 255), 2)
                    cv2.imwrite(f"output_images/selected_box_slice_{slice_idx}.jpg", slice_with_box)
                    print(f"已将selected_box绘制到切片{slice_idx}图像并保存")
                    
                    # 在原图上绘制矩形
                    if self.original_image is not None:
                        original_with_box = self.original_image.copy()
                        cv2.rectangle(original_with_box, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.imwrite("output_images/selected_box_original.jpg", original_with_box)
                        print("已将selected_box绘制到原图并保存")
                    break
        #计算x_croped的值，到这为止，x_croped的值已经计算出来了
        #-------------------------------------------------------------
        
        # 如果没有找到selected_box，设置默认的x_croped值
        if 'x_croped' not in locals():
            x_croped = None
            print("警告: 未找到合适的框，x_croped设置为None")
             
        index = 0
        for slice_info in slices_info:
            slice_img = slice_info['slice']
            slice_index = slice_info['slice_index']
            start_y = slice_info['start_y']
            
            print(f"处理切片 {slice_index}...")
            
            # 进行OCR识别
            slice_img_rgb = cv2.cvtColor(slice_img, cv2.COLOR_BGR2RGB)
            result = self.engine(slice_img_rgb)
            result.vis(f"output_images/slice_ocr_result_{index}.jpg")
            
            pass
            
            # 过滤低置信度结果
            if result.boxes is not None and result.txts is not None:
                filtered_boxes = []
                filtered_txts = []
                filtered_scores = []
                
                for box, txt, score in zip(result.boxes, result.txts, result.scores):
                    if score >= self.text_score_threshold:
                        filtered_boxes.append(box)
                        filtered_txts.append(txt)
                        filtered_scores.append(score)
                
                print(f"切片 {slice_index} 过滤后结果: {[(txt, score) for txt, score in zip(filtered_txts, filtered_scores)]}")
                
                if not filtered_boxes:
                    print(f"切片 {slice_index} 过滤后无有效文本")
                    # 即使没有文本，也创建空的切片结果
                    slice_result = {
                        'slice_index': slice_index,
                        'start_y': start_y,
                        'end_y': slice_info['end_y'],
                        'ocr_result': {
                            'boxes': [],
                            'txts': [],
                            'scores': [],
                            'image_shape': slice_img.shape
                        },
                        'avatar_positions': [],
                        'chat_result': None
                    }
                    slice_results.append(slice_result)
                    print(f"切片 {slice_index} (无文本) 添加到结果列表")
                    continue
                
                # 转换坐标到原图坐标系
                adjusted_boxes = []
                for box in filtered_boxes:
                    # box 格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    adjusted_box = []
                    for point in box:
                        adjusted_point = [point[0], point[1] + start_y]
                        adjusted_box.append(adjusted_point)
                    adjusted_boxes.append(adjusted_box)
                
                # 构建该切片的OCR结果，过滤后的ocr结果都保存到这里了
                slice_ocr_result = {
                    'boxes': adjusted_boxes,
                    'txts': filtered_txts,
                    'scores': filtered_scores,
                    'image_shape': slice_img.shape
                }
                
                #-------------------------------------------------------------
                # 对头像裁图进行处理，目的是找到头像的坐标
                print(f"分析切片 {slice_index} 的聊天消息...")
                # slice_chat_result = self.analyze_slice_chat_messages(slice_ocr_result, slice_img, start_y)
                # 如果x_croped为None，使用原图像；否则进行裁剪
                if x_croped is not None:
                    slice_img = slice_img[0:slice_img.shape[0],0:x_croped]
                    print(f"切片 {slice_index} 使用x_croped={x_croped}进行裁剪")
                else:
                    print(f"切片 {slice_index} 未进行x裁剪，使用原始图像")
                cv2.imwrite(f"./debug_images/slice_{slice_index:03d}_avatar.jpg", slice_img)
                #输入的是slice_img的左侧头像截图 得到所有头像的外接矩形的坐标 xmin ymin w h
                sliced_merged_result = process_avatar_v2(slice_img)
                # INSERT_YOUR_CODE
                # 将sliced_merged_result里的坐标还原到原图（y坐标加上start_y）
                if sliced_merged_result:
                    restored_sliced_merged_result = []
                    for (x, y, w, h) in sliced_merged_result:
                        restored_box = (x, y + start_y, w, h)
                        restored_sliced_merged_result.append(restored_box)
                    sliced_merged_result = restored_sliced_merged_result

                # INSERT_YOUR_CODE
                # 画红色框：sliced_merged_result
                # for (x, y, w, h) in sliced_merged_result:
                #     cv2.rectangle(slice_img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色BGR

                # # 画绿色框：slice_ocr_result['boxes']
                # for box in slice_ocr_result['boxes']:
                #     # box: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                #     pts = [(int(pt[0]), int(pt[1] - start_y)) for pt in box]  # 转回切片坐标
                #     for i in range(4):
                #         pt1 = pts[i]
                #         pt2 = pts[(i + 1) % 4]
                #         cv2.line(slice_img, pt1, pt2, (0, 255, 0), 2)  # 绿色BGR
                # cv2.imwrite(f"./debug/slice_{slice_index:03d}_avatar.jpg", slice_img)

                # INSERT_YOUR_CODE
                # sliced_merged_result从上到下排序
                if sliced_merged_result:
                    # 按y坐标升序排序（即从上到下）
                    sliced_merged_result = sorted(sliced_merged_result, key=lambda rect: rect[1])
                #到这为止，sliced_merged_result已经排序，并且是按照从上到下的顺序
                #-------------------------------------------------------------



                #-------------------------------------------------------------
                #到这里，ocr的结果有了 头像的坐标也有了
                # INSERT_YOUR_CODE
                # 将slice_ocr_result中的boxes按从上到下排序，txts和scores同步
                if slice_ocr_result['boxes']:
                    # 获取每个box的最小y坐标（即最上方的点）
                    box_with_index = []
                    for idx, box in enumerate(slice_ocr_result['boxes']):
                        min_y = min(pt[1] for pt in box)
                        box_with_index.append((min_y, idx, box))
                    # 按min_y升序排序
                    box_with_index.sort()
                    # 重新排列boxes, txts, scores
                    sorted_boxes = []
                    sorted_txts = []
                    sorted_scores = []
                    for _, idx, box in box_with_index:
                        sorted_boxes.append(slice_ocr_result['boxes'][idx])
                        sorted_txts.append(slice_ocr_result['txts'][idx])
                        sorted_scores.append(slice_ocr_result['scores'][idx])
                    slice_ocr_result['boxes'] = sorted_boxes
                    slice_ocr_result['txts'] = sorted_txts
                    slice_ocr_result['scores'] = sorted_scores

                # 将当前切片的OCR结果汇总到原图坐标系统
                for idx, box in enumerate(slice_ocr_result['boxes']):
                    ocr_item_original = {
                        'slice_index': slice_index,
                        'box': box,  # 已经是原图坐标
                        'text': slice_ocr_result['txts'][idx],
                        'score': slice_ocr_result['scores'][idx]
                    }
                    all_ocr_results_original.append(ocr_item_original)
                
                # 将当前切片的头像位置汇总到原图坐标系统
                if sliced_merged_result is not None:
                    for avatar_box in sliced_merged_result:
                        x, y, w, h = avatar_box
                        avatar_item_original = {
                            'slice_index': slice_index,
                            'box': (x, y, w, h),  # 已经是原图坐标
                            'center_x': x + w/2,
                            'center_y': y + h/2
                        }
                        all_avatar_positions_original.append(avatar_item_original)
                
                # 创建并添加切片结果到slice_results
                slice_result = {
                    'slice_index': slice_index,
                    'start_y': start_y,
                    'end_y': slice_info['end_y'],
                    'ocr_result': slice_ocr_result,
                    'avatar_positions': sliced_merged_result if sliced_merged_result else [],
                    'chat_result': None  # 现在基于去重后数据统一处理，这里暂时为None
                }
                slice_results.append(slice_result)
                print(f"切片 {slice_index} 处理完成，添加到结果列表")
            else:
                print(f"切片 {slice_index} 未检测到文本")
                # 创建空的切片结果
                slice_result = {
                    'slice_index': slice_index,
                    'start_y': start_y,
                    'end_y': slice_info['end_y'],
                    'ocr_result': {
                        'boxes': [],
                        'txts': [],
                        'scores': [],
                        'image_shape': slice_img.shape
                    },
                    'avatar_positions': [],
                    'chat_result': None
                }
                slice_results.append(slice_result)
                print(f"切片 {slice_index} (无OCR结果) 添加到结果列表")
            index += 1
                
        
        # 去重处理
        print(f"\n=== 开始去重处理 ===")
        deduplicated_ocr, deduplicated_avatars = self._deduplicate_results(all_ocr_results_original, all_avatar_positions_original)
        
        # 基于去重后的数据重新标记
        print(f"\n=== 基于去重后数据重新标记 ===")
        marked_ocr_results = self._remark_content_with_deduplicated_data(deduplicated_ocr, deduplicated_avatars)
        
        # 保存汇总结果到类属性
        self.all_ocr_results_original = deduplicated_ocr
        self.all_avatar_positions_original = deduplicated_avatars
        self.marked_ocr_results_original = marked_ocr_results
        
        # 保存标记后的OCR结果到JSON文件
        self._export_marked_ocr_results()
        pass
        
        # 整理并导出结构化的聊天消息
        LLM_input = self._export_structured_chat_messages()
        
        # 输出汇总统计信息
        print(f"\n=== 去重后汇总统计 ===")
        print(f"原图坐标系统中的OCR结果总数: {len(deduplicated_ocr)} (去重前: {len(all_ocr_results_original)})")
        print(f"原图坐标系统中的头像位置总数: {len(deduplicated_avatars)} (去重前: {len(all_avatar_positions_original)})")
        print(f"标记后的OCR结果总数: {len(marked_ocr_results)}")
        
        # 按切片显示详细信息  
        processed_slices = set(item['slice_index'] for item in deduplicated_ocr)
        for slice_idx in sorted(processed_slices):
            ocr_count = len([item for item in deduplicated_ocr if item['slice_index'] == slice_idx])
            avatar_count = len([item for item in deduplicated_avatars if item['slice_index'] == slice_idx])
            print(f"切片 {slice_idx}: OCR结果 {ocr_count} 个, 头像位置 {avatar_count} 个")
        
        return LLM_input
    
    def generate_chat_result_from_marked_ocr(self, slice_ocr_result: Dict, slice_index: int, start_y_offset: int) -> Dict:
        """
        从标记的OCR结果生成聊天分析结果
        按顺序处理：昵称+连续内容为一组，时间单独一组
        
        Args:
            slice_ocr_result: 包含标记文本的OCR结果
            slice_index: 切片索引
            start_y_offset: 切片在原图中的Y轴偏移量
            
        Returns:
            聊天分析结果，包含messages列表
        """
        messages = []
        current_message = None
        message_id = 0
        
        print(f"\n=== 开始处理切片 {slice_index} ===")
        
        for i, txt in enumerate(slice_ocr_result['txts']):
            print(f"处理第{i+1}项: {txt}")
            
            # 获取当前文本框的位置信息
            current_box = slice_ocr_result['boxes'][i]
            current_y = float(sum(point[1] for point in current_box) / 4)  # 计算box中心Y坐标
            
            if "(昵称)" in txt:
                # 1. 先保存当前消息（如果有内容）
                if current_message and current_message.get('内容').strip():
                    messages.append(current_message)
                    print(f"  → 保存消息: {current_message['昵称']} - {current_message['内容'][:50]}...")
                
                # 2. 开始新消息
                message_id += 1
                nickname = txt.replace("(昵称)", "").strip()
                current_message = {
                    'message_id': message_id,
                    '昵称': nickname,
                    '内容': "",
                    'time': "",
                    '是否本人': nickname == "我",
                    'slice_index': slice_index,
                    'message_y': current_y
                }
                print(f"  → 新消息开始: 昵称='{nickname}', Y={current_y:.1f}")
                
            elif "(内容)" in txt or "(我的内容)" in txt:
                # 添加内容到当前消息
                content = txt.replace("(内容)", "").replace("(我的内容)", "").strip()
                
                if current_message is None:
                    # 没有昵称直接有内容，创建消息
                    message_id += 1
                    current_message = {
                        'message_id': message_id,
                        '昵称': "我" if "(我的内容)" in txt else "未知",
                        '内容': "",
                        'time': "",
                        '是否本人': "(我的内容)" in txt,
                        'slice_index': slice_index,
                        'message_y': current_y
                    }
                
                # 累加内容
                if current_message['内容']:
                    current_message['内容'] += " " + content
                else:
                    current_message['内容'] = content
                
                # 更新是否本人
                if "(我的内容)" in txt:
                    current_message['是否本人'] = True
                    current_message['昵称'] = "我"
                
                print(f"  → 添加内容: {content}")
                
            elif "(时间)" in txt:
                time_text = txt.replace("(时间)", "").strip()
                
                # 所有时间标记都按照纯时间处理
                # 1. 先保存当前消息（如果有内容）
                if current_message and current_message.get('内容').strip():
                    messages.append(current_message)
                    print(f"  → 保存消息: {current_message['昵称']} - {current_message['内容'][:50]}...")
                    current_message = None
                
                # 2. 时间单独成组
                message_id += 1
                time_message = {
                    'message_id': message_id,
                    '昵称': "",
                    '内容': "",
                    'time': time_text,
                    '是否本人': False,
                    'slice_index': slice_index,
                    'message_y': current_y
                }
                messages.append(time_message)
                print(f"  → 保存时间: {time_text}")
                
            else:
                # 处理未标记文本
                if current_message is None:
                    # 创建新消息
                    message_id += 1
                    current_message = {
                        'message_id': message_id,
                        '昵称': "未知",
                        '内容': "",
                        'time': "",
                        '是否本人': False,
                        'slice_index': slice_index,
                        'message_y': current_y
                    }
                
                # 添加未标记文本作为内容
                if current_message['内容']:
                    current_message['内容'] += " " + txt.strip()
                else:
                    current_message['内容'] = txt.strip()
                print(f"  → 添加未标记文本: {txt.strip()}")
        
        # 保存最后一条消息
        if current_message and current_message.get('内容').strip():
            messages.append(current_message)
            print(f"  → 保存最后消息: {current_message['昵称']} - {current_message['内容'][:50]}...")
        
        print(f"=== 切片 {slice_index} 处理完成，共生成 {len(messages)} 条消息 ===\n")
        
        # 输出最终整理结果
        print("【最终整理结果】")
        for i, msg in enumerate(messages, 1):
            if msg['time']:  # 时间消息
                print(f"{i}. [时间] {msg['time']}")
            else:  # 普通消息
                is_me = " (我)" if msg['是否本人'] else ""
                print(f"{i}. [消息] {msg['昵称']}{is_me}: {msg['内容']}")
        
        return {
            'total_messages': len(messages),
            'messages': messages
        }
    
    def _is_pure_time_format(self, text: str) -> bool:
        """
        判断文本是否为纯时间格式
        
        Args:
            text: 要检查的文本
            
        Returns:
            是否为纯时间格式
        """
        import re
        
        # 纯时间格式的正则表达式
        pure_time_patterns = [
            r'^\d{4}年\d{1,2}月\d{1,2}日\d{1,2}[：:]\d{2}$',  # 2025年6月23日18:22
            r'^\d{1,2}月\d{1,2}日\s*\d{1,2}[：:]\d{2}$',      # 7月14日 11:53
            r'^(今天|昨天|前天|明天)\s*\d{1,2}[：:]\d{2}$',    # 今天 11:53
            r'^(上午|下午|凌晨|早上|中午|晚上)\s*\d{1,2}[：:]\d{2}$',  # 上午11:53
            r'^\d{1,2}[：:]\d{2}([：:]\d{2})?$',              # 11:53, 11:53:30
            r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}$',                # 2023-07-14
            r'^\d{1,2}/\d{1,2}$',                            # 7/14
        ]
        
        text = text.strip()
        
        # 如果文本很长（超过30个字符），很可能不是纯时间
        if len(text) > 30:
            return False
        
        # 检查是否匹配纯时间格式
        for pattern in pure_time_patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def merge_chat_results(self, slice_results: List[Dict]) -> Dict:
        """
        合并各个切片的聊天分析结果，基于box位置去除重叠区域的重复消息
        
        Args:
            slice_results: 切片结果列表，每个包含ocr_result和chat_result
            
        Returns:
            merged_chat_result: 合并后的聊天分析结果
        """
        all_messages = []
        
        # 1. 收集所有切片的消息，保留位置信息
        for slice_result in slice_results:
            slice_index = slice_result['slice_index']
            chat_result = slice_result['chat_result']
            
            for message in chat_result['messages']:
                # 复制消息并设置切片信息，保留位置信息
                merged_message = message.copy()
                merged_message['slice_index'] = slice_index
                
                # 如果消息没有位置信息，尝试从其他字段推断
                if 'message_y' not in merged_message:
                    # 可以根据切片位置和消息在切片中的位置来估算
                    merged_message['message_y'] = slice_result.get('start_y', 0) + message.get('message_id', 0) * 50
                    
                all_messages.append(merged_message)
        
        print(f"合并前共有 {len(all_messages)} 条消息")
        
        # 2. 按照Y坐标位置排序所有消息（而不是切片索引）
        all_messages.sort(key=lambda x: (x.get('message_y', 0), x['slice_index']))
        
        # 3. 基于位置的智能去重：去除相邻切片间位置重叠的重复内容
        filtered_messages = []
        
        for current_msg in all_messages:
            is_duplicate = False
            
            # 检查已过滤消息中位置相近的消息
            for existing_msg in filtered_messages:
                # 基于位置的重复检测
                if self._is_position_based_duplicate(current_msg, existing_msg):
                    is_duplicate = True
                    print(f"跳过位置重复: 切片{current_msg['slice_index']} Y={current_msg.get('message_y', 0):.1f} - '{current_msg['内容'][:30]}...'")
                    break
            
            if not is_duplicate:
                filtered_messages.append(current_msg)
        
        print(f"去重后共有 {len(filtered_messages)} 条消息")
        
        # 4. 生成最终结果
        final_messages = []
        for i, msg in enumerate(filtered_messages):
            formatted_message = {
                'message_id': i + 1,
                '昵称': msg['昵称'],
                '内容': msg['内容'],
                '时间': msg['time'],
                '是否本人': msg['是否本人'],
                'slice_index': int(msg['slice_index'])
            }
            final_messages.append(formatted_message)
        
        return {
            'total_messages': len(final_messages),
            'messages': final_messages
        }
    
    def _is_simple_overlap_duplicate(self, msg1: Dict, msg2: Dict) -> bool:
        """
        简单的重叠重复检测：专门用于检测切片重叠区域的重复
        
        Args:
            msg1, msg2: 要比较的消息
            
        Returns:
            是否为重叠重复
        """
        # 只检查相邻切片
        slice_diff = abs(msg1['slice_index'] - msg2['slice_index'])
        
        if slice_diff > 1:
            return False
        
        # 内容相同（只在跨切片时检查）
        content1 = msg1['内容'].strip() if msg1['内容'] else ""
        content2 = msg2['内容'].strip() if msg2['内容'] else ""
        
        if (slice_diff >= 1 and  # 只在跨切片时检查内容重复
            content1 and content2 and content1 == content2):
            return True
        
        # 昵称和时间都相同（只在跨切片时检查）
        if (slice_diff >= 1 and  # 只在跨切片时检查昵称重复
            msg1['昵称'] == msg2['昵称'] and 
            msg1['time'] == msg2['time'] and 
            msg1['昵称'] != ""):
            return True
        
        return False
    
    def _is_position_based_duplicate(self, msg1: Dict, msg2: Dict) -> bool:
        """
        基于位置、昵称和内容的精确重复检测：只有完全相同才删除
        
        Args:
            msg1, msg2: 要比较的消息
            
        Returns:
            是否为重复
        """
        # 1. 检查是否来自相邻切片
        slice_diff = abs(msg1['slice_index'] - msg2['slice_index'])
        if slice_diff > 1:
            return False
        
        # 2. 获取位置信息
        y1 = msg1.get('message_y', 0)
        y2 = msg2.get('message_y', 0)
        
        # 3. 获取内容和昵称
        nickname1 = msg1['昵称'].strip() if msg1['昵称'] else ""
        nickname2 = msg2['昵称'].strip() if msg2['昵称'] else ""
        content1 = msg1['内容'].strip() if msg1['内容'] else ""
        content2 = msg2['内容'].strip() if msg2['内容'] else ""
        
        # 4. 严格匹配：昵称、内容和message_y都相同才认为是重复
        is_duplicate = (
            abs(y1 - y2) < 20 and  # 位置几乎相同(允许微小误差)
            nickname1 == nickname2 and  # 昵称完全相同
            content1 == content2  # 内容完全相同
        )
        
        if is_duplicate:
            print(f"   -> 严格重复检测: 昵称='{nickname1}', Y位置相差={abs(y1 - y2):.1f}像素, 内容相同")
            
        return is_duplicate
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度
        
        Args:
            text1, text2: 要比较的文本
            
        Returns:
            相似度 (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        # 简单的字符集合交集方法
        set1 = set(text1)
        set2 = set(text2)
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _is_duplicate_message(self, msg1: Dict, msg2: Dict) -> bool:
        """
        判断两条消息是否重复
        
        Args:
            msg1, msg2: 要比较的消息
            
        Returns:
            是否重复
        """
        # 提取基本信息
        content1 = msg1['内容'].strip() if msg1['内容'] else ""
        content2 = msg2['内容'].strip() if msg2['内容'] else ""
        nickname1 = msg1['昵称'].strip() if msg1['昵称'] else ""
        nickname2 = msg2['昵称'].strip() if msg2['昵称'] else ""
        time1 = msg1['time'].strip() if msg1['time'] else ""
        time2 = msg2['time'].strip() if msg2['time'] else ""
        
        # 检查是否来自不同切片
        slice_diff = abs(msg1['slice_index'] - msg2['slice_index'])
        
        # 1. 特殊情况：消息内容被错分为昵称（最高优先级检测）
        # 情况1：msg1有内容，msg2没内容但昵称是msg1的内容
        special_case1 = content1 and not content2 and nickname2 == content1
        # 情况2：msg2有内容，msg1没内容但昵称是msg2的内容  
        special_case2 = content2 and not content1 and nickname1 == content2
        # 情况3：昵称包含对方的内容（部分匹配）
        special_case3 = content1 and not content2 and content1 in nickname2 and len(content1) > 3
        special_case4 = content2 and not content1 and content2 in nickname1 and len(content2) > 3
        
        if special_case1 or special_case2 or special_case3 or special_case4:
            # 对于特殊情况，允许跨切片重复（最多相邻2个切片）
            if slice_diff <= 2:
                print(f"    检测到特殊重复情况：消息内容被错误分为昵称 (切片差距: {slice_diff})")
                return True
        
        # 2. 内容完全相同的重复（同昵称且内容相同）
        if content1 and content2 and content1 == content2:
            # 如果昵称也相同，强烈怀疑重复
            if nickname1 == nickname2:
                # 允许跨相邻切片
                if slice_diff <= 1:
                    print(f"    检测到完全相同的重复消息 (切片差距: {slice_diff})")
                    return True
            # 如果昵称不同但内容完全相同，可能是同一人发了两条相同消息
            # 只在同一切片内认为是重复
            elif slice_diff == 0:
                print(f"    检测到内容相同但昵称不同的可能重复 (切片差距: {slice_diff})")
                return True
        
        # 3. 昵称相似且内容相关的情况
        if self._are_nicknames_similar(nickname1, nickname2) and slice_diff <= 1:
            # 如果昵称相似，且一个有内容另一个没有，可能是分割错误
            if (content1 and not content2) or (content2 and not content1):
                print(f"    检测到昵称相似且内容不对称的重复情况 (切片差距: {slice_diff})")
                return True
        
        # 4. 内容相似度检测（在合理切片范围内）
        if content1 and content2 and slice_diff <= 1:
            # 去除标点符号后比较
            clean_content1 = re.sub(r'[^\w\u4e00-\u9fff]', '', content1)
            clean_content2 = re.sub(r'[^\w\u4e00-\u9fff]', '', content2)
            
            if clean_content1 == clean_content2 and clean_content1:
                print(f"    检测到去除标点后内容相同的重复 (切片差距: {slice_diff})")
                return True
            
            # 计算相似度
            if clean_content1 and clean_content2:
                similarity = self._calculate_text_similarity(clean_content1, clean_content2)
                
                # 如果相似度很高（>85%），认为是重复
                if similarity > 0.85:
                    print(f"    检测到高相似度重复消息 (相似度: {similarity:.2f}, 切片差距: {slice_diff})")
                    return True
                
                # 如果一个内容包含另一个，且长度比例合理
                if clean_content1 in clean_content2 or clean_content2 in clean_content1:
                    min_len = min(len(clean_content1), len(clean_content2))
                    max_len = max(len(clean_content1), len(clean_content2))
                    if min_len > 0 and (min_len / max_len) > 0.8:  # 80%包含度
                        print(f"    检测到包含关系的重复消息 (包含度: {min_len/max_len:.2f}, 切片差距: {slice_diff})")
                        return True
        
        # 5. 空内容消息的重复检测
        if not content1 and not content2 and slice_diff == 0:
            # 两个都是空内容且在同一切片
            if time1 == time2 and time1:  # 时间相同
                print(f"    检测到时间相同的空内容重复 (切片差距: {slice_diff})")
                return True
            if nickname1 == nickname2:  # 昵称相同
                print(f"    检测到昵称相同的空内容重复 (切片差距: {slice_diff})")
                return True
        
        return False
    
    def _are_nicknames_similar(self, nickname1: str, nickname2: str) -> bool:
        """
        判断两个昵称是否相似
        
        Args:
            nickname1, nickname2: 要比较的昵称
            
        Returns:
            是否相似
        """
        if not nickname1 or not nickname2:
            return False
        
        # 去除特殊字符后比较
        clean_nick1 = re.sub(r'[^\w\u4e00-\u9fff]', '', nickname1)
        clean_nick2 = re.sub(r'[^\w\u4e00-\u9fff]', '', nickname2)
        
        if clean_nick1 == clean_nick2:
            return True
        
        # 检查一个是否包含另一个
        if clean_nick1 and clean_nick2:
            if clean_nick1 in clean_nick2 or clean_nick2 in clean_nick1:
                min_len = min(len(clean_nick1), len(clean_nick2))
                max_len = max(len(clean_nick1), len(clean_nick2))
                if min_len > 0 and (min_len / max_len) > 0.8:  # 80%相似度
                    return True
        
        return False
    
    def _compare_message_quality(self, msg1: Dict, msg2: Dict) -> int:
        """
        比较两条消息的质量
        
        Args:
            msg1, msg2: 要比较的消息
            
        Returns:
            1: msg1质量更高, -1: msg2质量更高, 0: 质量相当
        """
        score1 = 0
        score2 = 0
        
        # 1. 内容长度（更长且有实际内容的更好）
        content1 = msg1['内容'].strip()
        content2 = msg2['内容'].strip()
        
        # 有内容比没有内容好
        if content1 and not content2:
            score1 += 5
        elif content2 and not content1:
            score2 += 5
        elif content1 and content2:
            # 都有内容时，比较长度
            len1 = len(content1)
            len2 = len(content2)
            if len1 > len2:
                score1 += 2
            elif len2 > len1:
                score2 += 2
        
        # 2. 昵称信息（明确的昵称比"未知"好）
        nickname1 = msg1['昵称'].strip()
        nickname2 = msg2['昵称'].strip()
        
        if nickname1 and nickname1 != "未知" and (not nickname2 or nickname2 == "未知"):
            score1 += 3
        elif nickname2 and nickname2 != "未知" and (not nickname1 or nickname1 == "未知"):
            score2 += 3
        
        # 3. 时间信息（有时间信息的更好）
        time1 = msg1['time'].strip() if msg1['time'] else ""
        time2 = msg2['time'].strip() if msg2['time'] else ""
        
        if time1 and not time2:
            score1 += 2
        elif time2 and not time1:
            score2 += 2
        
        # 4. 消息ID偏好（ID更小的是先处理的，可能更准确）
        if msg1['message_id'] < msg2['message_id']:
            score1 += 1
        elif msg2['message_id'] < msg1['message_id']:
            score2 += 1
        
        # 5. 切片索引偏好（在重叠区域，倾向于保留来自后面切片的消息，因为可能更完整）
        if msg1['slice_index'] > msg2['slice_index']:
            score1 += 1
        elif msg2['slice_index'] > msg1['slice_index']:
            score2 += 1
        
        if score1 > score2:
            return 1
        elif score2 > score1:
            return -1
        else:
            return 0

    def merge_results(self, slice_results: List[Dict], original_shape: Tuple[int, int, int]) -> Dict:
        """
        整合OCR结果，处理重叠区域
        
        Args:
            slice_results: 切片结果列表，每个包含ocr_result和chat_result
            original_shape: 原始图像形状 (h, w, c)
            
        Returns:
            merged_result: 整合后的结果
        """
        # 从slice_results中提取OCR结果
        ocr_results = []
        for slice_result in slice_results:
            ocr_result = {
                'slice_index': slice_result['slice_index'],
                'start_y': slice_result['start_y'],
                'end_y': slice_result['end_y'],
                'boxes': slice_result['ocr_result']['boxes'],
                'txts': slice_result['ocr_result']['txts'],
                'scores': slice_result['ocr_result']['scores']
            }
            ocr_results.append(ocr_result)
        
        if not ocr_results:
            return {'boxes': [], 'txts': [], 'scores': []}
        
        merged_boxes = []
        merged_txts = []
        merged_scores = []
        
        for i, result in enumerate(ocr_results):
            current_boxes = result['boxes']
            current_txts = result['txts']
            current_scores = result['scores']
            current_start_y = result['start_y']
            current_end_y = result['end_y']
            
            for j, (box, txt, score) in enumerate(zip(current_boxes, current_txts, current_scores)):
                # 检查是否在重叠区域
                box_top_y = min(point[1] for point in box)
                box_bottom_y = max(point[1] for point in box)
                
                # 判断是否需要过滤重叠区域的文本
                should_keep = True
                duplicate_index = -1  # 记录重复项的索引
                
                if i > 0:  # 不是第一个切片
                    # 检查是否在重叠区域内
                    overlap_start = current_start_y
                    overlap_end = current_start_y + self.overlap
                    
                    # 如果文本框主要在重叠区域内，则检查是否与前面的结果重复
                    if box_top_y >= overlap_start and box_top_y < overlap_end:
                        # 在重叠区域内，检查是否与已有结果重复
                        for k, (existing_box, existing_txt, existing_score) in enumerate(zip(merged_boxes, merged_txts, merged_scores)):
                            if self._is_duplicate_text(box, txt, existing_box, existing_txt):
                                # 发现重复，比较置信度
                                if score > existing_score:
                                    # 当前文本框置信度更高，删除之前的，保留当前的
                                    duplicate_index = k
                                    should_keep = True
                                    print(f"发现重复文本 - 当前: '{txt}'({score:.3f}) vs 已存在: '{existing_txt}'({existing_score:.3f}), 保留置信度更高的: '{txt}'")
                                else:
                                    # 之前的文本框置信度更高，跳过当前的
                                    should_keep = False
                                    print(f"发现重复文本 - 当前: '{txt}'({score:.3f}) vs 已存在: '{existing_txt}'({existing_score:.3f}), 保留置信度更高的: '{existing_txt}'")
                                break
                
                if should_keep:
                    # 如果需要删除之前的重复项
                    if duplicate_index >= 0:
                        # 删除置信度较低的重复项
                        del merged_boxes[duplicate_index]
                        del merged_txts[duplicate_index]
                        del merged_scores[duplicate_index]
                    
                    # 添加当前项
                    merged_boxes.append(box)
                    merged_txts.append(txt)
                    merged_scores.append(score)
        
        return {
            'boxes': merged_boxes,
            'txts': merged_txts,
            'scores': merged_scores,
            'image_shape': original_shape
        }
    
    def analyze_slice_chat_messages(self, slice_ocr_result: Dict, slice_image: np.ndarray, start_y_offset: int) -> Dict:
        """
        分析单个切片的聊天消息，按位置和内容进行分类整理
        
        Args:
            slice_ocr_result: 切片的OCR结果
            slice_image: 切片图像
            start_y_offset: 切片在原图中的Y轴偏移量
            
        Returns:
            分析后的聊天消息结构
        """
        if not slice_ocr_result['boxes'] or not slice_ocr_result['txts']:
            return {'messages': []}
        
        # 获取所有文本框的信息
        text_boxes = []
        for i, (box, txt, score) in enumerate(zip(
            slice_ocr_result['boxes'], 
            slice_ocr_result['txts'], 
            slice_ocr_result['scores']
        )):
            # 计算文本框的中心点和边界（相对于原图坐标）
            center_x = float(np.mean([p[0] for p in box]))
            center_y = float(np.mean([p[1] for p in box]))
            min_x = float(min([p[0] for p in box]))
            max_x = float(max([p[0] for p in box]))
            min_y = float(min([p[1] for p in box]))
            max_y = float(max([p[1] for p in box]))
            
            # 检测文本框区域的颜色（需要将坐标转换回切片坐标系）
            slice_box = [[p[0], p[1] - start_y_offset] for p in box]
            is_green_box = self._detect_green_content_box(slice_image, slice_box)
            
            text_boxes.append({
                'id': i,
                'text': txt,
                'score': score,
                'box': box,
                'center_x': center_x,
                'center_y': center_y,
                'min_x': min_x,
                'max_x': max_x,
                'min_y': min_y,
                'max_y': max_y,
                'width': float(max_x - min_x),
                'height': float(max_y - min_y),
                'is_green_box': is_green_box  # 是否为绿色框
            })
        
        # 1. 使用DBSCAN在Y轴上进行聚类，分成不同的消息组
        print(f"  使用DBSCAN在Y轴上进行消息分组...")
        y_coordinates = np.array([[tb['center_y']] for tb in text_boxes])
        
        # 对于切片，使用固定的eps值（切片高度1200，设置为36像素，约3%）
        slice_eps = 40
        print(f"  切片高度: {self.slice_height}, 使用eps值: {slice_eps}")

        # 使用固定eps值
        dbscan = DBSCAN(eps=slice_eps, min_samples=2)
        cluster_labels = dbscan.fit_predict(y_coordinates)
        
        # 将文本框按聚类分组
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(text_boxes[i])
        
        print(f"  共分为 {len(clusters)} 个消息组")
        
        # 2. 对每个聚类组进行分析
        messages = []
        
        for cluster_id, cluster_boxes in clusters.items():
            if cluster_id == -1:  # 噪声点，跳过
                continue
                
            print(f"  分析消息组 {cluster_id}，包含 {len(cluster_boxes)} 个文本框")
            
            # 按center_y排序，确保消息的垂直顺序
            cluster_boxes.sort(key=lambda x: x['center_y'])
            
            # 3. 根据颜色和内容特征分类文本框
            green_boxes = []      # 绿色内容框（本人消息）
            time_boxes = []       # 时间戳框
            left_boxes = []       # 左侧框（昵称、头像、其他人内容）
            
            # 时间模式匹配
            time_pattern = r'\d{1,2}:\d{2}|\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日|上午|下午|\d{1,2}:\d{2}:\d{2}'
            
            for box in cluster_boxes:
                # 首先检查是否为绿色框
                if box['is_green_box']:
                    green_boxes.append(box)
                    print(f"    检测到绿色框: '{box['text']}'")
                # 然后检查是否为时间戳
                elif re.search(time_pattern, box['text']):
                    time_boxes.append(box)
                    print(f"    检测到时间框: '{box['text']}'")
                # 其余的归为左侧框
                else:
                    left_boxes.append(box)
                    print(f"    归类为左侧框: '{box['text']}'")
            
            # 4. 判断消息类型和提取信息
            is_self_message = len(green_boxes) > 0  # 有绿色框则为本人消息
            
            # 提取时间
            time_text = ""
            if time_boxes:
                # 选择最符合时间格式的文本
                time_candidates = []
                for box in time_boxes:
                    if re.search(r'\d{1,2}:\d{2}', box['text']):
                        time_candidates.append(box)
                
                if time_candidates:
                    time_text = time_candidates[0]['text']
                else:
                    time_text = time_boxes[0]['text']
            
            # 5. 提取昵称和内容
            nickname = ""
            content_texts = []
            
            if is_self_message:
                # 本人消息：昵称为"本人"，内容来自绿色框
                nickname = "我"
                content_texts = [box['text'] for box in sorted(green_boxes, key=lambda x: (x['center_y'], x['center_x']))]
            else:
                # 他人消息：从左侧框中区分昵称和内容
                if left_boxes:
                    # 过滤掉可能是头像的框
                    potential_nicknames = []
                    potential_contents = []
                    potential_avatars = []
                    
                    for box in left_boxes:
                        text = box['text'].strip()
                        
                        # 判断是否可能是头像框（文本很短、包含特殊符号、尺寸很小）
                        # 只有包含特殊符号才可能是头像框
                        if text in ['□', '○', '◯', '●', '▲', '■',"®"]:
                            potential_avatars.append(box)
                        # 判断是否可能是昵称（较短的有意义文本，位置靠上）
                        elif len(text) <= 10 and re.search(r'[\u4e00-\u9fff\w]', text):
                            potential_nicknames.append(box)
                        # 其余的可能是内容
                        else:
                            potential_contents.append(box)
                    
                    # 选择昵称（优先选择位置最靠上且最短的有意义文本）
                    if potential_nicknames:
                        nickname_box = min(potential_nicknames, key=lambda x: (x['center_y'], len(x['text'])))
                        nickname = nickname_box['text']
                        
                        # 从潜在内容中移除昵称框
                        remaining_boxes = [box for box in left_boxes if box['id'] != nickname_box['id']]
                        content_texts = [box['text'] for box in sorted(remaining_boxes, key=lambda x: (x['center_y'], x['center_x']))]
                    else:
                        # 没有明确的昵称，将所有非头像框的文本作为内容
                        content_texts = [box['text'] for box in sorted(potential_contents, key=lambda x: (x['center_y'], x['center_x']))]
                        nickname = "未知"
            
            # 6. 整理消息
            if content_texts or nickname != "未知":  # 至少要有内容或有效昵称才算一条消息
                content = ' '.join(content_texts) if content_texts else ""
                
                message = {
                    'cluster_id': cluster_id,
                    '昵称': nickname,
                    '内容': content,
                    'time': time_text,
                    '是否本人': is_self_message,
                    'message_y': float(min(box['center_y'] for box in cluster_boxes)),
                    'components': {
                        'green_boxes_count': len(green_boxes),
                        'time_boxes_count': len(time_boxes),
                        'left_boxes_count': len(left_boxes)
                    }
                }
                messages.append(message)
                
                print(f"    -> 昵称: {message['昵称']}")
                print(f"    -> 内容: {message['内容'][:50]}...")
                print(f"    -> 时间: {message['time']}")
                print(f"    -> 本人消息: {message['是否本人']}")
        
        # 7. 按从上到下的顺序排序消息
        messages.sort(key=lambda x: x['message_y'])
        
        return {
            'total_messages': len(messages),
            'messages': messages
        }

    def analyze_chat_messages(self, merged_result: Dict, original_image: np.ndarray = None) -> Dict:
        """
        分析聊天消息，按位置和内容进行分类整理
        
        Args:
            merged_result: 整合后的OCR结果
            original_image: 原始图像，用于颜色检测
            
        Returns:
            分析后的聊天消息结构
        """
        if not merged_result['boxes'] or not merged_result['txts']:
            return {'messages': []}
        
        # 获取所有文本框的信息
        text_boxes = []
        for i, (box, txt, score) in enumerate(zip(
            merged_result['boxes'], 
            merged_result['txts'], 
            merged_result['scores']
        )):
            # 计算文本框的中心点和边界
            center_x = float(np.mean([p[0] for p in box]))
            center_y = float(np.mean([p[1] for p in box]))
            min_x = float(min([p[0] for p in box]))
            max_x = float(max([p[0] for p in box]))
            min_y = float(min([p[1] for p in box]))
            max_y = float(max([p[1] for p in box]))
            
            # 检测文本框区域的颜色
            is_green_box = False
            if original_image is not None:
                is_green_box = self._detect_green_content_box(original_image, box)
            
            text_boxes.append({
                'id': i,
                'text': txt,
                'score': score,
                'box': box,
                'center_x': center_x,
                'center_y': center_y,
                'min_x': min_x,
                'max_x': max_x,
                'min_y': min_y,
                'max_y': max_y,
                'width': float(max_x - min_x),
                'height': float(max_y - min_y),
                'is_green_box': is_green_box  # 是否为绿色框
            })
        
        # 1. 使用DBSCAN在Y轴上进行聚类，分成不同的消息组
        print("步骤1: 使用DBSCAN在Y轴上进行消息分组...")
        y_coordinates = np.array([[tb['center_y']] for tb in text_boxes])
        
        # 计算自适应eps值
        image_height = original_image.shape[0]
        adaptive_eps =  int(image_height * 0.003) # 图片高度的1.5%，最小值为30
        print(f"图片高度: {image_height}, 自适应eps值: {adaptive_eps}")

        # 使用自适应eps值
        dbscan = DBSCAN(eps=adaptive_eps, min_samples=1)
        cluster_labels = dbscan.fit_predict(y_coordinates)
        
        # 将文本框按聚类分组
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(text_boxes[i])
        
        print(f"共分为 {len(clusters)} 个消息组")
        
        # 2. 对每个聚类组进行分析
        messages = []
        
        for cluster_id, cluster_boxes in clusters.items():
            if cluster_id == -1:  # 噪声点，跳过
                continue
                
            print(f"分析消息组 {cluster_id}，包含 {len(cluster_boxes)} 个文本框")
            
            # 按center_y排序，确保消息的垂直顺序
            cluster_boxes.sort(key=lambda x: x['center_y'])
            
            # 3. 根据颜色和内容特征分类文本框
            green_boxes = []      # 绿色内容框（本人消息）
            time_boxes = []       # 时间戳框
            left_boxes = []       # 左侧框（昵称、头像、其他人内容）
            
            # 时间模式匹配
            time_pattern = r'\d{1,2}:\d{2}|\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}|\d{1,2}月\d{1,2}日|上午|下午|\d{1,2}:\d{2}:\d{2}'
            
            for box in cluster_boxes:
                # 首先检查是否为绿色框
                if box['is_green_box']:
                    green_boxes.append(box)
                    print(f"  检测到绿色框: '{box['text']}'")
                # 然后检查是否为时间戳
                elif re.search(time_pattern, box['text']):
                    time_boxes.append(box)
                    print(f"  检测到时间框: '{box['text']}'")
                # 其余的归为左侧框
                else:
                    left_boxes.append(box)
                    print(f"  归类为左侧框: '{box['text']}'")
            
            # 4. 判断消息类型和提取信息
            is_self_message = len(green_boxes) > 0  # 有绿色框则为本人消息
            
            # 提取时间
            time_text = ""
            if time_boxes:
                # 选择最符合时间格式的文本
                time_candidates = []
                for box in time_boxes:
                    if re.search(r'\d{1,2}:\d{2}', box['text']):
                        time_candidates.append(box)
                
                if time_candidates:
                    time_text = time_candidates[0]['text']
                else:
                    time_text = time_boxes[0]['text']
            
            # 5. 提取昵称和内容
            nickname = ""
            content_texts = []
            
            if is_self_message:
                # 本人消息：昵称为"本人"，内容来自绿色框
                nickname = "我"
                content_texts = [box['text'] for box in sorted(green_boxes, key=lambda x: (x['center_y'], x['center_x']))]
            else:
                # 他人消息：从左侧框中区分昵称和内容
                if left_boxes:
                    # 过滤掉可能是头像的框
                    potential_nicknames = []
                    potential_contents = []
                    potential_avatars = []
                    
                    for box in left_boxes:
                        text = box['text'].strip()
                        
                        # 判断是否可能是头像框（文本很短、包含特殊符号、尺寸很小）
                        # 只有包含特殊符号才可能是头像框
                        if text in ['□', '○', '◯', '●', '▲', '■',"®"]:
                            potential_avatars.append(box)
                        # 判断是否可能是昵称（较短的有意义文本，位置靠上）
                        elif len(text) <= 10 and re.search(r'[\u4e00-\u9fff\w]', text):
                            potential_nicknames.append(box)
                        # 其余的可能是内容
                        else:
                            potential_contents.append(box)
                    
                    # 选择昵称（优先选择位置最靠上且最短的有意义文本）
                    if potential_nicknames:
                        nickname_box = min(potential_nicknames, key=lambda x: (x['center_y'], len(x['text'])))
                        nickname = nickname_box['text']
                        
                        # 从潜在内容中移除昵称框
                        remaining_boxes = [box for box in left_boxes if box['id'] != nickname_box['id']]
                        content_texts = [box['text'] for box in sorted(remaining_boxes, key=lambda x: (x['center_y'], x['center_x']))]
                    else:
                        # 没有明确的昵称，将所有非头像框的文本作为内容
                        content_texts = [box['text'] for box in sorted(potential_contents, key=lambda x: (x['center_y'], x['center_x']))]
                        nickname = "未知"
            
            # 6. 整理消息
            if content_texts or nickname != "未知":  # 至少要有内容或有效昵称才算一条消息
                content = ' '.join(content_texts) if content_texts else ""
                
                message = {
                    'cluster_id': cluster_id,
                    '昵称': nickname,
                    '内容': content,
                    'time': time_text,
                    '是否本人': is_self_message,
                    'message_y': float(min(box['center_y'] for box in cluster_boxes)),
                    'components': {
                        'green_boxes_count': len(green_boxes),
                        'time_boxes_count': len(time_boxes),
                        'left_boxes_count': len(left_boxes)
                    }
                }
                messages.append(message)
                
                print(f"  -> 昵称: {message['昵称']}")
                print(f"  -> 内容: {message['内容'][:50]}...")
                print(f"  -> 时间: {message['time']}")
                print(f"  -> 本人消息: {message['是否本人']}")
        
        # 7. 按从上到下的顺序排序消息
        messages.sort(key=lambda x: x['message_y'])
        
        # 8. 生成最终结果
        result = {
            'total_messages': len(messages),
            'messages': []
        }
        
        for i, msg in enumerate(messages):
            formatted_message = {
                'message_id': i + 1,
                '昵称': msg['昵称'],
                '内容': msg['内容'],
                '时间': msg['time'],
                '是否本人': msg['是否本人']
            }
            result['messages'].append(formatted_message)
        
        return result
    
    def save_chat_analysis_result(self, chat_result: Dict, image_path: str) -> str:
        """
        保存聊天分析结果
        
        Args:
            chat_result: 聊天分析结果
            image_path: 原始图像路径
            
        Returns:
            保存的JSON文件路径
        """
        # 保存详细的聊天分析结果
        image_name = Path(image_path).stem
        json_path = self.output_json_dir / f"{image_name}_chat_analysis.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(chat_result, f, ensure_ascii=False, indent=2)
        
        print(f"保存聊天分析结果: {json_path}")
        return str(json_path)
    
    def _is_duplicate_text(self, box1: List, txt1: str, box2: List, txt2: str, 
                          threshold: float = 50.0, iou_threshold: float = 0.5) -> bool:
        """
        判断两个文本框是否重复
        
        Args:
            box1, txt1: 第一个文本框和文本
            box2, txt2: 第二个文本框和文本
            threshold: 中心点距离阈值
            iou_threshold: IOU阈值
            
        Returns:
            是否重复
        """
        # 计算两个框的中心点距离
        center1 = np.mean(box1, axis=0)
        center2 = np.mean(box2, axis=0)
        distance = np.linalg.norm(center1 - center2)
        
        # 计算IOU
        iou = self._calculate_iou(box1, box2)
        
        # 如果位置距离太远且IOU很小，肯定不重复
        if distance > threshold and iou < 0.1:
            return False
        
        # 位置接近或有一定重叠的情况下，检查文本内容关系
        # 1. 完全相同
        if txt1 == txt2:
            # 如果文本完全相同，只要有一定重叠就认为是重复
            return iou > 0.1 or distance < threshold
        
        # 2. 去除空格和标点后比较
        clean_txt1 = re.sub(r'[^\w\u4e00-\u9fff]', '', txt1)
        clean_txt2 = re.sub(r'[^\w\u4e00-\u9fff]', '', txt2)
        
        if clean_txt1 == clean_txt2 and clean_txt1:  # 确保不为空
            return iou > 0.1 or distance < threshold
        
        # 3. 检查包含关系（一个文本是另一个的子串）
        if clean_txt1 and clean_txt2:
            # 如果一个文本完全包含另一个，且长度差不太大，认为是重复
            if clean_txt1 in clean_txt2 or clean_txt2 in clean_txt1:
                # 计算文本长度比例，如果差异不大，认为是重复
                min_len = min(len(clean_txt1), len(clean_txt2))
                max_len = max(len(clean_txt1), len(clean_txt2))
                if min_len > 0 and (min_len / max_len) > 0.6:  # 较短文本至少是较长文本的60%
                    # 包含关系的情况下，需要更高的IOU或更近的距离
                    return iou > iou_threshold or distance < threshold * 0.8
        
        # 4. 检查文本相似度（Levenshtein距离）
        if clean_txt1 and clean_txt2:
            similarity = self._calculate_text_similarity(clean_txt1, clean_txt2)
            if similarity > 0.8:  # 相似度大于80%认为是重复
                # 高相似度的情况下，需要有一定的重叠
                return iou > iou_threshold or distance < threshold * 0.7
        
        # 5. 高IOU的情况下，即使文本稍有不同也可能是重复
        if iou > 0.4:  # 高重叠度
            # 检查是否是相似的短文本
            if clean_txt1 and clean_txt2 and max(len(clean_txt1), len(clean_txt2)) <= 10:
                similarity = self._calculate_text_similarity(clean_txt1, clean_txt2)
                if similarity > 0.5:  # 对于短文本，降低相似度要求
                    return True
        
        return False
    
    def _calculate_iou(self, box1: List, box2: List) -> float:
        """
        计算两个文本框的IOU（Intersection over Union）
        
        Args:
            box1, box2: 文本框坐标，格式为[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            
        Returns:
            IOU值，0-1之间的浮点数
        """
        try:
            # 将四边形转换为轴对齐的边界框
            def get_bbox(box):
                x_coords = [point[0] for point in box]
                y_coords = [point[1] for point in box]
                return {
                    'x_min': min(x_coords),
                    'y_min': min(y_coords),
                    'x_max': max(x_coords),
                    'y_max': max(y_coords)
                }
            
            bbox1 = get_bbox(box1)
            bbox2 = get_bbox(box2)
            
            # 计算交集
            x_left = max(bbox1['x_min'], bbox2['x_min'])
            y_top = max(bbox1['y_min'], bbox2['y_min'])
            x_right = min(bbox1['x_max'], bbox2['x_max'])
            y_bottom = min(bbox1['y_max'], bbox2['y_max'])
            
            # 检查是否有交集
            if x_right <= x_left or y_bottom <= y_top:
                return 0.0
            
            # 计算交集面积
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            
            # 计算各自的面积
            area1 = (bbox1['x_max'] - bbox1['x_min']) * (bbox1['y_max'] - bbox1['y_min'])
            area2 = (bbox2['x_max'] - bbox2['x_min']) * (bbox2['y_max'] - bbox2['y_min'])
            
            # 计算并集面积
            union_area = area1 + area2 - intersection_area
            
            # 计算IOU
            if union_area <= 0:
                return 0.0
            
            iou = intersection_area / union_area
            return max(0.0, min(1.0, iou))  # 确保在0-1范围内
            
        except Exception as e:
            # 如果计算出错，返回0
            print(f"计算IOU时出错: {e}")
            return 0.0
    
    def _calculate_text_similarity(self, str1: str, str2: str) -> float:
        """
        计算两个字符串的相似度（基于编辑距离）
        
        Args:
            str1, str2: 要比较的字符串
            
        Returns:
            相似度，0-1之间的浮点数
        """
        if not str1 or not str2:
            return 0.0
        
        # 计算编辑距离
        len1, len2 = len(str1), len(str2)
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        # 初始化
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        
        # 填充dp表
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        
        # 计算相似度
        max_len = max(len1, len2)
        if max_len == 0:
            return 1.0
        
        edit_distance = dp[len1][len2]
        similarity = 1 - (edit_distance / max_len)
        return max(0.0, similarity)
    
    def save_json_result(self, merged_result: Dict, image_path: str) -> str:
        """
        保存JSON结果文件
        
        Args:
            merged_result: 整合后的结果
            image_path: 原始图像路径
            
        Returns:
            保存的JSON文件路径
        """
        # 转换numpy数组为列表，便于JSON序列化
        json_result = {
            'image_path': image_path,
            'image_shape': merged_result['image_shape'],
            'total_texts': len(merged_result['txts']),
            'results': []
        }
        
        for i, (box, txt, score) in enumerate(zip(
            merged_result['boxes'], 
            merged_result['txts'], 
            merged_result['scores']
        )):
            text_item = {
                'id': i,
                'text': txt,
                'confidence': float(score),
                'box': [[float(p[0]), float(p[1])] for p in box],
                'center': [
                    float(np.mean([p[0] for p in box])),
                    float(np.mean([p[1] for p in box]))
                ]
            }
            json_result['results'].append(text_item)
        
        # 保存JSON文件
        image_name = Path(image_path).stem
        json_path = self.output_json_dir / f"{image_name}_ocr_result.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, ensure_ascii=False, indent=2)
        
        print(f"保存JSON结果: {json_path}")
        return str(json_path)
    
    def visualize_final_result(self, original_image: np.ndarray, merged_result: Dict, 
                             image_path: str) -> str:
        """
        可视化最终结果
        
        Args:
            original_image: 原始图像
            merged_result: 整合后的结果
            image_path: 原始图像路径
            
        Returns:
            可视化图像保存路径
        """
        from PIL import Image, ImageDraw, ImageFont
        
        # 转换为PIL图像
        if len(original_image.shape) == 3:
            # BGR to RGB
            pil_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(original_image)
        
        # 创建绘图对象
        draw = ImageDraw.Draw(pil_image)
        
        # 尝试加载中文字体
        try:
            # 在fonts文件夹中查找字体文件
            font_path = Path("fonts/SourceHanSansCN-Regular.otf")
            if font_path.exists():
                font = ImageFont.truetype(str(font_path), size=20)
            else:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # 绘制文本框和文本
        colors = [(0, 255, 0)]
        
        for i, (box, txt, score) in enumerate(zip(
            merged_result['boxes'], 
            merged_result['txts'], 
            merged_result['scores']
        )):
            color = colors[i % len(colors)]
            
            # 绘制文本框
            box_points = [(int(p[0]), int(p[1])) for p in box]
            draw.polygon(box_points, outline=color, width=2)
            
            # 绘制文本
            text_pos = (int(box[0][0]), max(0, int(box[0][1]) - 25))
            draw.text(text_pos, f"{txt} ({score:.2f})", fill=color, font=font)
        
        # 保存可视化结果
        image_name = Path(image_path).stem
        vis_path = self.output_images_dir / f"{image_name}_final_result.jpg"
        
        # 转换回OpenCV格式并保存
        final_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(vis_path), final_img)
        
        print(f"保存最终可视化结果: {vis_path}")
        return str(vis_path)
    
    def process_long_image(self, image_path: str) -> Dict:
        """
        处理长图的完整流程
        
        Args:
            image_path: 图像路径
            
        Returns:
            处理结果摘要
        """
        print(f"开始处理长图: {image_path}")
        
        # 1. 切分图像
        print("步骤1: 切分图像...")
        original_image, slices_info = self.slice_image(image_path)
        # 保存原图到实例属性中，供其他方法使用
        self.original_image = original_image
        print(f"共切分为 {len(slices_info)} 个切片")
        
        # 2. OCR处理和切片级聊天分析
        print("步骤2: OCR处理和切片级聊天分析...")
        LLMs_input = self.process_slices(slices_info)
        print(f"LLMs_input: {LLMs_input["chat_messages"]}")
      

        #将结果输入到ollama模型
        print("步骤7: 将结果输入到ollama模型...")
        while True:
            user_question = input("请输入你想问的问题（输入'退出'结束）：")
            if user_question.strip() in ["退出", "q", "Q", "exit"]:
                print("已退出与ollama模型的交互。")
                break
            process_with_llm(user_question, LLMs_input["chat_messages"])

       

    def _detect_green_content_box(self, image: np.ndarray, box: List) -> bool:
        """
        检测文本框区域是否为绿色背景（本人消息框）
        
        Args:
            image: 原始图像
            box: 文本框坐标
            
        Returns:
            是否为绿色框
        """
        try:
            # 获取文本框区域
            points = np.array(box, dtype=np.int32)
            min_x = max(0, int(np.min(points[:, 0])))
            max_x = min(image.shape[1], int(np.max(points[:, 0])))
            min_y = max(0, int(np.min(points[:, 1])))
            max_y = min(image.shape[0], int(np.max(points[:, 1])))
            
            if max_x <= min_x or max_y <= min_y:
                return False
            
            # 提取区域图像
            roi = image[min_y:max_y, min_x:max_x]
            # 保存ROI到本地，用于调试
            debug_dir = Path("./output_images/debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成唯一文件名，使用时间戳和随机数
            import time
            import random
            timestamp = int(time.time() * 1000)
            random_num = random.randint(1000, 9999)
            roi_filename = f"roi_{timestamp}_{random_num}.jpg"
            
            # 保存ROI图像
            roi_path = debug_dir / roi_filename
            cv2.imwrite(str(roi_path), roi)
            print(f"已保存ROI图像到: {roi_path}")
            
            if roi.size == 0:
                return False
            
            # 转换为HSV颜色空间进行绿色检测
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # 定义绿色的HSV范围
            # 浅绿色范围（聊天界面中常见的绿色）
            lower_green1 = np.array([35, 40, 40])
            upper_green1 = np.array([85, 255, 255])
            
            # 创建绿色掩码
            mask = cv2.inRange(hsv, lower_green1, upper_green1)
            
            # 计算绿色像素的比例
            green_pixels = cv2.countNonZero(mask)
            total_pixels = roi.shape[0] * roi.shape[1]
            
            if total_pixels > 0:
                green_ratio = green_pixels / total_pixels
                # 如果绿色像素超过20%，认为是绿色框
                return green_ratio > 0.2
            
            return False
            
        except Exception as e:
            print(f"检测绿色框时出错: {e}")
            return False
    
    def _detect_blue_content_box(self, image: np.ndarray, box: List) -> bool:
        """
        检测文本框区域是否为蓝色背景（本人消息框）
        
        Args:
            image: 原始图像
            box: 文本框坐标
            
        Returns:
            是否为蓝色框
        """
        try:
            # 获取文本框区域
            points = np.array(box, dtype=np.int32)
            min_x = max(0, int(np.min(points[:, 0])))
            max_x = min(image.shape[1], int(np.max(points[:, 0])))
            min_y = max(0, int(np.min(points[:, 1])))
            max_y = min(image.shape[0], int(np.max(points[:, 1])))
            
            if max_x <= min_x or max_y <= min_y:
                return False
            
            # 提取区域图像
            roi = image[min_y:max_y, min_x:max_x]
            
            if roi.size == 0:
                return False
            
            # 转换为HSV颜色空间进行蓝色检测
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # 定义蓝色的HSV范围（聊天气泡常见的淡蓝色）
            lower_blue = np.array([100, 30, 80])    # 降低饱和度下限，提高亮度下限
            upper_blue = np.array([130, 180, 255])  # 降低饱和度上限，避免深蓝色文字
            
            # 创建蓝色掩码
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # 额外检查：排除白色背景上的蓝色文字
            # 检测白色背景
            lower_white = np.array([0, 0, 200])     # 白色HSV范围
            upper_white = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # 计算像素比例
            blue_pixels = cv2.countNonZero(blue_mask)
            white_pixels = cv2.countNonZero(white_mask)
            total_pixels = roi.shape[0] * roi.shape[1]
            
            if total_pixels > 0:
                blue_ratio = blue_pixels / total_pixels
                white_ratio = white_pixels / total_pixels
                
                # 判断逻辑：
                # 1. 蓝色比例要足够高(>30%)
                # 2. 白色比例不能太高(<50%)，避免白底蓝字的情况
                # 3. 蓝色比例要大于白色比例，确保是蓝底而不是白底
                is_blue_background = (blue_ratio > 0.3 and 
                                    white_ratio < 0.5 and 
                                    blue_ratio > white_ratio)
                
                # print(f"蓝色检测 - 蓝色比例: {blue_ratio:.3f}, 白色比例: {white_ratio:.3f}, 判断为蓝色背景: {is_blue_background}")
                return is_blue_background
            
            return False
            
        except Exception as e:
            print(f"检测蓝色框时出错: {e}")
            return False
    
    def _calculate_box_iou(self, box1, box2):
        """
        计算两个矩形框的IoU (Intersection over Union)
        
        Args:
            box1, box2: 可以是两种格式
                - OCR box: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                - Avatar box: (x, y, w, h)
        
        Returns:
            float: IoU值 (0-1)
        """
        try:
            # 处理OCR box格式 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            if isinstance(box1[0], list):
                x1_min = min(pt[0] for pt in box1)
                y1_min = min(pt[1] for pt in box1)
                x1_max = max(pt[0] for pt in box1)
                y1_max = max(pt[1] for pt in box1)
            else:  # 处理Avatar box格式 (x, y, w, h)
                x1_min, y1_min, w1, h1 = box1
                x1_max = x1_min + w1
                y1_max = y1_min + h1
            
            if isinstance(box2[0], list):
                x2_min = min(pt[0] for pt in box2)
                y2_min = min(pt[1] for pt in box2)
                x2_max = max(pt[0] for pt in box2)
                y2_max = max(pt[1] for pt in box2)
            else:  # 处理Avatar box格式 (x, y, w, h)
                x2_min, y2_min, w2, h2 = box2
                x2_max = x2_min + w2
                y2_max = y2_min + h2
            
            # 计算交集
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)
            
            if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
                return 0.0
            
            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            
            # 计算并集
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)
            union_area = area1 + area2 - inter_area
            
            return inter_area / union_area if union_area > 0 else 0.0
            
        except Exception as e:
            print(f"计算IoU时出错: {e}")
            return 0.0
    
    def _deduplicate_results(self, ocr_results, avatar_positions):
        """
        对OCR结果和头像位置进行去重
        
        Args:
            ocr_results: 原始OCR结果列表
            avatar_positions: 原始头像位置列表
            
        Returns:
            tuple: (去重后的OCR结果, 去重后的头像位置)
        """
        print("开始OCR结果去重...")
        deduplicated_ocr = []
        ocr_iou_threshold = 0.65  # OCR去重IoU阈值
        
        for i, current_ocr in enumerate(ocr_results):
            is_duplicate = False
            current_box = current_ocr['box']
            
            # 与已添加的OCR结果比较
            for existing_ocr in deduplicated_ocr:
                existing_box = existing_ocr['box']
                iou = self._calculate_box_iou(current_box, existing_box)
                
                if iou > ocr_iou_threshold:
                    is_duplicate = True
                    # 保留置信度更高的结果
                    if current_ocr['score'] > existing_ocr['score']:
                        # 替换现有结果
                        idx = deduplicated_ocr.index(existing_ocr)
                        deduplicated_ocr[idx] = current_ocr
                        print(f"替换重复OCR (IoU={iou:.3f}): '{existing_ocr['text']}' -> '{current_ocr['text']}'")
                    else:
                        print(f"跳过重复OCR (IoU={iou:.3f}): '{current_ocr['text']}'")
                    break
            
            if not is_duplicate:
                deduplicated_ocr.append(current_ocr)
        
        print(f"OCR去重完成: {len(ocr_results)} -> {len(deduplicated_ocr)}")
        
        print("开始头像位置去重...")
        deduplicated_avatars = []
        avatar_iou_threshold = 0.0  # 头像去重IoU阈值
        
        for i, current_avatar in enumerate(avatar_positions):
            current_box = current_avatar['box']
            current_area = current_box[2] * current_box[3]  # w * h
            
            # 检查是否与已存在的头像重复
            duplicate_index = -1
            for j, existing_avatar in enumerate(deduplicated_avatars):
                existing_box = existing_avatar['box']
                iou = self._calculate_box_iou(current_box, existing_box)
                
                if iou > avatar_iou_threshold:
                    duplicate_index = j
                    existing_area = existing_box[2] * existing_box[3]  # w * h
                    
                    if current_area > existing_area:
                        # 当前头像面积更大，替换已存在的
                        print(f"替换重复头像 (IoU={iou:.3f}): slice_{existing_avatar['slice_index']} (面积={existing_area}) -> slice_{current_avatar['slice_index']} (面积={current_area})")
                        deduplicated_avatars[j] = current_avatar
                    else:
                        # 已存在的头像面积更大，跳过当前头像
                        print(f"跳过重复头像 (IoU={iou:.3f}): slice_{current_avatar['slice_index']} (面积={current_area}) vs slice_{existing_avatar['slice_index']} (面积={existing_area})")
                    break
            
            # 如果没有重复，直接添加
            if duplicate_index == -1:
                deduplicated_avatars.append(current_avatar)
        
        print(f"头像去重完成: {len(avatar_positions)} -> {len(deduplicated_avatars)}")
        
        return deduplicated_ocr, deduplicated_avatars
    
    def _remark_content_with_deduplicated_data(self, deduplicated_ocr, deduplicated_avatars):
        """
        基于去重后的数据重新标记内容
        
        Args:
            deduplicated_ocr: 去重后的OCR结果
            deduplicated_avatars: 去重后的头像位置
            
        Returns:
            list: 标记后的OCR结果
        """
        print("开始重新标记...")
        
        # 创建工作副本，避免修改原始数据
        marked_results = []
        for ocr_item in deduplicated_ocr:
            marked_item = ocr_item.copy()
            marked_item['original_text'] = marked_item['text']  # 保存原始文本
            marked_results.append(marked_item)
        
        # 按Y坐标排序头像和OCR结果
        sorted_avatars = sorted(deduplicated_avatars, key=lambda x: x['center_y'])
        sorted_ocr = sorted(marked_results, key=lambda x: self._get_box_center_y(x['box']))
        
        print(f"处理 {len(sorted_avatars)} 个头像和 {len(sorted_ocr)} 个OCR结果")
        
        # 1. 首先标记时间
        self._mark_time_content(sorted_ocr)
        
        # 2. 基于头像位置标记昵称和内容
        self._mark_nickname_and_content_with_avatars(sorted_ocr, sorted_avatars)
        
        # 重新排序，因为可能插入了虚拟昵称
        sorted_ocr.sort(key=lambda x: self._get_box_center_y(x['box']))
        print(f"插入虚拟昵称后，OCR结果总数: {len(sorted_ocr)}")
        
        # 3. 标记绿色背景的内容为"我的内容"
        self._mark_green_content(sorted_ocr, sorted_avatars)
        
        print("重新标记完成")
        return sorted_ocr  # 返回处理后的sorted_ocr，包含虚拟昵称
    
    def _get_box_y_min(self, box):
        """获取box的中心Y坐标"""
        if isinstance(box[0], list):  # OCR box格式
            # return sum(pt[1] for pt in box) / 4
            return box[3][1]
        else:  # Avatar box格式 (x, y, w, h)
            return box[1] + box[3] / 2
    
    def _get_box_center_y(self, box):
        """获取box的中心Y坐标"""
        if isinstance(box[0], list):  # OCR box格式
            return sum(pt[1] for pt in box) / 4
            # return box[3][1]
        else:  # Avatar box格式 (x, y, w, h)
            return box[1] + box[3] / 2
            
    def _mark_time_content(self, ocr_results):
        """标记时间内容（使用严格的时间检测条件）"""
        import re
        
        # 时间模式列表
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
        
        for ocr_item in ocr_results:
            text = ocr_item['text'].strip()
            
            # 排除过长的文本（超过30个字符的很可能不是纯时间）
            if len(text) > 30:
                continue
                
            # 排除包含明显非时间关键词的文本
            exclude_keywords = ['报送', '回执', '会议', '参加', '人员', '工作', '通知', '安排', '要求', '地点', '内容', 
                              '完成', '需要', '前', '后', '开始', '结束', '传包', '表格', '填写', '更新', '自测']
            if any(keyword in text for keyword in exclude_keywords):
                continue
            
            # 检查时间模式
            is_time = False
            for pattern in time_patterns:
                if re.search(pattern, text):
                    match = re.search(pattern, text)
                    if match:
                        matched_length = len(match.group())
                        match_ratio = matched_length / len(text)
                        
                        # 对于复合时间格式（如"昨天晚上6:23"），降低阈值要求
                        if pattern.startswith('(昨天|今天|前天|明天)') or pattern.startswith('(上午|下午|早上|中午|晚上|凌晨)'):
                            # 复合时间格式，匹配40%以上即可
                            if match_ratio >= 0.4:
                                is_time = True
                                break
                        else:
                            # 简单时间格式，仍然要求60%以上
                            if match_ratio >= 0.6:
                                is_time = True
                                break
            
            if is_time:
                ocr_item['text'] = text + "(时间)"
                print(f"标记时间: {ocr_item['text']}")
            else:
                print(f"跳过非纯时间文本: {text[:50]}...")
    
    def _mark_nickname_and_content_with_avatars(self, ocr_results, avatars):
        """基于头像位置标记昵称和内容"""
        # 一次性绘制所有头像框（只在第一次调用时执行）
        if self.original_image is not None and not hasattr(self, '_avatars_drawn'):
            debug_img = self.original_image.copy()
            # 画出所有头像框
            for idx, avatar_item in enumerate(avatars):
                avatar_box_ = avatar_item['box']
                x, y, w, h = avatar_box_
                # 在原图上绘制矩形，使用红色(0,0,255)
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # 在框旁边添加索引编号
                cv2.putText(debug_img, f"{idx}:y={y}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 保存带有头像框的图像
            output_path = "output_images/avatars_marked_1.jpg"
            cv2.imwrite(output_path, debug_img)
            print(f"已将所有头像框绘制到原图并保存至 {output_path}")
            self._avatars_drawn = True
        elif self.original_image is None:
            print("原图不可用，无法绘制头像框")
        
        # 收集需要插入的虚拟昵称
        virtual_nicknames_to_insert = []
        
        for i, avatar in enumerate(avatars):
            avatar_box = avatar['box']
            x_min, y_min, w, h = avatar_box
            y_max = y_min + h
            
            # 找到下一个头像的Y位置边界
            next_boundary = avatars[i+1]['box'][1] if i+1 < len(avatars) else float('inf')
            
            # 寻找在当前头像Y范围内的文本作为昵称
            nickname_found = False
            nickname_start_index = -1
            
            for j, ocr_item in enumerate(ocr_results):
                if "(时间)" in ocr_item['text']:  # 跳过已标记的时间
                    continue
                    
                box_y_min = self._get_box_y_min(ocr_item['box'])
                
                # 检查是否在头像Y范围内
                if y_min <= box_y_min <= y_max and not nickname_found:
                    # 标记为昵称
                    ocr_item['text'] = ocr_item['text'] + "(昵称)"
                    print(f"标记昵称: {ocr_item['text']}")
                    nickname_found = True
                    nickname_start_index = j
                    break
            
            # 如果没有找到昵称，记录需要插入虚拟昵称
            if not nickname_found:
                print(f"警告: 头像 {i} (y={y_min}-{y_max}) 附近未找到昵称，准备插入虚拟昵称")
                
                # 找到插入位置
                insert_index = len(ocr_results)  # 默认插入到末尾
                for idx, ocr_item in enumerate(ocr_results):
                    item_y_min = self._get_box_y_min(ocr_item['box'])
                    if item_y_min > y_max:
                        insert_index = idx
                        break
                
                # 创建虚拟昵称条目
                virtual_nickname = {
                    'text': f"未知用户{i+1}(昵称)",
                    'box': [[x_min, y_min], [x_min + w, y_min], [x_min + w, y_max], [x_min, y_max]],
                    'confidence': 0.0,
                    'slice_index': avatar.get('slice_index', -1),
                    'virtual': True,
                    'insert_index': insert_index  # 记录插入位置
                }
                
                virtual_nicknames_to_insert.append(virtual_nickname)
                print(f"准备插入虚拟昵称: {virtual_nickname['text']} 在位置 {insert_index}")
        
        # 统一插入虚拟昵称（按插入位置倒序插入，避免位置偏移）
        virtual_nicknames_to_insert.sort(key=lambda x: x['insert_index'], reverse=True)
        for virtual_nickname in virtual_nicknames_to_insert:
            insert_index = virtual_nickname['insert_index']
            del virtual_nickname['insert_index']  # 删除临时字段
            ocr_results.insert(insert_index, virtual_nickname)
            print(f"已插入虚拟昵称: {virtual_nickname['text']} 在位置 {insert_index}")
        
        # 现在标记内容（重新遍历，因为可能插入了虚拟昵称）
        for i, avatar in enumerate(avatars):
            avatar_box = avatar['box']
            x_min, y_min, w, h = avatar_box
            y_max = y_min + h
            
            # 找到下一个头像的Y位置边界
            next_boundary = avatars[i+1]['box'][1] if i+1 < len(avatars) else float('inf')
            
            # 找到对应的昵称
            nickname_index = -1
            for j, ocr_item in enumerate(ocr_results):
                if "(昵称)" in ocr_item['text']:
                    box_y_min = self._get_box_y_min(ocr_item['box'])
                    if y_min <= box_y_min <= y_max:
                        nickname_index = j
                        break
            
            if nickname_index >= 0:
                # 标记该昵称后的内容
                for k in range(nickname_index + 1, len(ocr_results)):
                    next_ocr = ocr_results[k]
                    if "(时间)" in next_ocr['text'] or "(昵称)" in next_ocr['text']:  # 跳过时间和昵称
                        continue
                    
                    next_box_y_min = self._get_box_y_min(next_ocr['box'])
                    
                    # 检查是否在当前头像区域内且未到达下一个头像边界
                    if next_box_y_min > y_min and next_box_y_min < next_boundary:
                        if "(内容)" not in next_ocr['text']:  # 避免重复标记
                            next_ocr['text'] = next_ocr['text'] + "(内容)"
                            print(f"标记内容: {next_ocr['text']}")
                    elif next_box_y_min >= next_boundary:
                        break
    
    def _mark_green_content(self, ocr_results, avatar_positions=None):
        """标记绿色和蓝色背景的内容（基于颜色检测和位置推理）"""
        if self.original_image is None:
            print("原图不可用，跳过颜色内容检测")
            return
        
        # 第一轮：基于颜色检测标记我的内容
        my_content_boxes = []  # 存储已确认的我的内容框
        
        for i, ocr_item in enumerate(ocr_results):
            if "(内容)" in ocr_item['text']:
                box = ocr_item['box']
                
                # 基于绿色背景检测
                is_green = False
                try:
                    is_green = self._detect_green_content_box(self.original_image, box)
                except Exception as e:
                    print(f"绿色检测失败: {e}")
                
                # 基于蓝色背景检测
                is_blue = False
                try:
                    is_blue = self._detect_blue_content_box(self.original_image, box)
                except Exception as e:
                    print(f"蓝色检测失败: {e}")
                
                # 绿色或蓝色背景都标记为"我的内容"
                if is_green:
                    ocr_item['text'] = ocr_item['text'].replace("(内容)", "(我的内容)")
                    my_content_boxes.append({'index': i, 'box': box})
                    print(f"标记为我的内容: {ocr_item['text']} (原因: 绿色背景)")
                elif is_blue:
                    ocr_item['text'] = ocr_item['text'].replace("(内容)", "(我的内容)")
                    my_content_boxes.append({'index': i, 'box': box})
                    print(f"标记为我的内容: {ocr_item['text']} (原因: 蓝色背景)")
        
        # 第二轮：基于位置推理标记相邻的内容
        self._mark_adjacent_my_content(ocr_results, my_content_boxes, avatar_positions)
    
    def _mark_adjacent_my_content(self, ocr_results, my_content_boxes, avatar_positions=None):
        """
        基于位置推理标记相邻的我的内容
        
        逻辑：如果通过颜色检测确定该条是我的内容，如果下一条：
        1. 没有通过颜色检测
        2. 处于最近的两个头像框之间（y1max --- y2min）
        则下一条也是我的内容
        """
        # 使用传入的avatar_positions，如果没有则使用类属性
        avatars = avatar_positions if avatar_positions is not None else self.all_avatar_positions_original
        
        print(f"开始位置推理：有 {len(my_content_boxes)} 个我的内容框，{len(avatars)} 个头像")
        
        if not my_content_boxes or not avatars:
            print(f"跳过位置推理：my_content_boxes={len(my_content_boxes) if my_content_boxes else 0}, avatars={len(avatars) if avatars else 0}")
            return
        
        for my_content in my_content_boxes:
            my_index = my_content['index']
            my_box = my_content['box']
            my_x_min = self._get_box_x_min(my_box)
            my_y_max = self._get_box_y_max(my_box)
            
            print(f"处理我的内容[{my_index}]: '{ocr_results[my_index]['text'][:50]}...', x_min={my_x_min}, y_max={my_y_max}")
            
            # 查找下一条内容
            for next_index in range(my_index + 1, len(ocr_results)):
                next_item = ocr_results[next_index]
                
                # 跳过已经标记为我的内容的
                if "(我的内容)" in next_item['text']:
                    print(f"  跳过[{next_index}]: 已是我的内容 - '{next_item['text'][:50]}...'")
                    continue
                
                # 只处理内容标记
                if "(内容)" not in next_item['text']:
                    print(f"  跳过[{next_index}]: 不是内容标记 - '{next_item['text'][:50]}...'")
                    continue
                
                next_box = next_item['box']
                next_x_min = self._get_box_x_min(next_box)
                next_y_min = self._get_box_y_min(next_box)
                
                print(f"  检查[{next_index}]: '{next_item['text'][:50]}...', x_min={next_x_min}, y_min={next_y_min}")
                
                # 检查是否已通过颜色检测（如果已通过，跳过）
                try:
                    is_green = self._detect_green_content_box(self.original_image, next_box)
                    is_blue = self._detect_blue_content_box(self.original_image, next_box)
                    if is_green or is_blue:
                        print(f"    跳过: 已通过颜色检测 (绿色={is_green}, 蓝色={is_blue})")
                        continue  # 已通过颜色检测，跳过
                except Exception as e:
                    print(f"    颜色检测异常: {e}")
                    pass
                
                # 检查位置条件
                print(f"    检查位置条件...")
                if self._is_adjacent_my_content(my_box, next_box, avatars):
                    next_item['text'] = next_item['text'].replace("(内容)", "(我的内容)")
                    print(f"✅ 基于位置推理标记为我的内容: {next_item['text']} (xmin对齐且在头像范围内)")
                    # 将新标记的内容也加入列表，以便继续检查下一条
                    my_content_boxes.append({'index': next_index, 'box': next_box})
                else:
                    print(f"    ❌ 位置条件不满足")
    
    def _is_adjacent_my_content(self, my_box, next_box, avatars):
        """
        检查下一条内容是否应该标记为我的内容
        
        条件：
        1. 处于最近的两个头像框之间
        （已取消X坐标对齐限制）
        """
        my_x_min = self._get_box_x_min(my_box)
        my_y_max = self._get_box_y_max(my_box)
        
        next_x_min = self._get_box_x_min(next_box)
        next_y_min = self._get_box_y_min(next_box)
        
        # 1. X坐标检查已取消 - 只要在头像范围内即可
        print(f"      X坐标信息: my_x={my_x_min}, next_x={next_x_min} (无限制)")
        
        # 2. 检查是否在最近的两个头像框之间
        print(f"      Y坐标检查: my_y_max={my_y_max}, next_y_min={next_y_min}")
        if not self._is_between_avatars(my_y_max, next_y_min, avatars):
            print(f"      ❌ 不在头像范围内")
            return False
        
        print(f"      ✅ 所有条件满足")
        return True
    
    def _is_between_avatars(self, start_y, end_y, avatars):
        """
        检查Y坐标范围是否在最近的两个头像框之间
        """
        if not avatars:
            return True  # 如果没有头像数据，允许通过
        
        # 按Y坐标排序头像
        sorted_avatars = sorted(avatars, 
                              key=lambda x: self._get_box_center_y(x['box']))
        
        # 找到包含start_y的头像对
        for i in range(len(sorted_avatars) - 1):
            avatar1 = sorted_avatars[i]
            avatar2 = sorted_avatars[i + 1]
            
            avatar1_y_max = self._get_box_y_max(avatar1['box'])
            avatar2_y_min = self._get_avatar_y_min(avatar2['box'])
            
            # 检查是否在这两个头像之间
            if avatar1_y_max <= start_y and end_y <= avatar2_y_min:
                return True
        
        return False
    
    def _get_box_x_min(self, box):
        """获取文本框的最小X坐标"""
        if isinstance(box, (tuple, list)) and len(box) == 4 and isinstance(box[0], (int, float)):
            # 头像格式: (x, y, w, h)
            return box[0]
        else:
            # OCR格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            return min(point[0] for point in box)
    
    def _get_box_y_max(self, box):
        """获取文本框的最大Y坐标（底部）"""
        if isinstance(box, (tuple, list)) and len(box) == 4 and isinstance(box[0], (int, float)):
            # 头像格式: (x, y, w, h)
            return box[1] + box[3]  # y + h
        else:
            # OCR格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            return max(point[1] for point in box)
    
    def _get_avatar_y_min(self, box):
        """获取头像框的最小Y坐标（顶部）"""
        if isinstance(box, (tuple, list)) and len(box) == 4 and isinstance(box[0], (int, float)):
            # 头像格式: (x, y, w, h)
            return box[1]  # y
        else:
            # 如果是其他格式，返回最小Y
            return min(point[1] for point in box)

    def get_summary_data(self) -> Dict:
        """
        获取汇总数据：原图坐标系统中的所有OCR结果、头像位置和标记后的OCR结果
        
        Returns:
            包含OCR结果、头像位置和标记后OCR结果的字典
        """
        return {
            'ocr_results_original': self.all_ocr_results_original,
            'avatar_positions_original': self.all_avatar_positions_original,
            'marked_ocr_results_original': self.marked_ocr_results_original,
            'statistics': {
                'total_ocr_items': len(self.all_ocr_results_original),
                'total_avatars': len(self.all_avatar_positions_original),
                'total_marked_ocr_items': len(self.marked_ocr_results_original),
                'processed_slices': len(set(item['slice_index'] for item in self.all_ocr_results_original)) if self.all_ocr_results_original else 0
            }
        }
    
    def export_summary_data(self, output_path: str = None) -> str:
        """
        导出汇总数据到JSON文件
        
        Args:
            output_path: 输出文件路径，如果为None则使用默认路径
            
        Returns:
            实际的输出文件路径
        """
        if output_path is None:
            output_path = self.output_json_dir / "summary_data_original.json"
        
        summary_data = self.get_summary_data()
        
        # 转换numpy类型为Python原生类型以便JSON序列化
        import json
        import numpy as np
        
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        # 递归转换所有numpy类型
        def deep_convert(data):
            if isinstance(data, dict):
                return {k: deep_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [deep_convert(v) for v in data]
            else:
                return convert_numpy(data)
        
        summary_data = deep_convert(summary_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        print(f"汇总数据已导出到: {output_path}")
        return str(output_path)
    
    def _export_marked_ocr_results(self, output_path: str = None) -> str:
        """
        导出标记后的OCR结果到JSON文件
        
        Args:
            output_path: 输出文件路径，如果为None则使用默认路径
            
        Returns:
            实际的输出文件路径
        """
        if output_path is None:
            output_path = self.output_json_dir / "marked_ocr_results_original.json"
        
        if not self.marked_ocr_results_original:
            print("没有标记后的OCR结果可导出")
            return ""
        
        import json
        from datetime import datetime
        
        # 提取所有text字段
        text_results = [item.get('text', '') for item in self.marked_ocr_results_original]
        
        export_data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_items": len(text_results),
                "description": "标记后的OCR文本结果 - 只包含文本内容"
            },
            "marked_texts": text_results
        }
        
        # 按类型分类统计
        time_count = len([text for text in text_results if "(时间)" in text])
        nickname_count = len([text for text in text_results if "(昵称)" in text])
        content_count = len([text for text in text_results if "(内容)" in text])
        my_content_count = len([text for text in text_results if "(我的内容)" in text])
        
        export_data["statistics"] = {
            "time_items": time_count,
            "nickname_items": nickname_count,
            "content_items": content_count,
            "my_content_items": my_content_count,
            "unmarked_items": len(text_results) - time_count - nickname_count - content_count - my_content_count
        }
        
        # 保存到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"标记后的OCR结果已导出到: {output_path}")
        print(f"  - 时间标记: {time_count} 项")
        print(f"  - 昵称标记: {nickname_count} 项") 
        print(f"  - 内容标记: {content_count} 项")
        print(f"  - 我的内容: {my_content_count} 项")
        print(f"  - 未标记: {export_data['statistics']['unmarked_items']} 项")
        
        return str(output_path)
    
    def _export_structured_chat_messages(self, output_path: str = None) -> str:
        """
        整理并导出结构化的聊天消息
        
        Args:
            output_path: 输出文件路径，如果为None则使用默认路径
            
        Returns:
            实际的输出文件路径
        """
        if output_path is None:
            output_path = self.output_json_dir / "structured_chat_messages.json"
        
        if not self.marked_ocr_results_original:
            print("没有标记后的OCR结果可整理")
            return ""
        
        import json
        from datetime import datetime
        
        # 提取所有标记后的文本
        marked_texts = [item.get('text', '') for item in self.marked_ocr_results_original]
        
        # 整理成结构化聊天消息
        structured_messages = self._organize_chat_messages(marked_texts)
        
        # 保存到类属性
        self.structured_chat_messages = structured_messages
        
        # 准备导出数据
        export_data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_messages": len(structured_messages),
                "description": "结构化聊天消息 - 按昵称、内容、时间组织"
            },
            "chat_messages": structured_messages
        }
        
        # 统计信息
        nickname_messages = len([msg for msg in structured_messages if msg.get('type') == 'chat'])
        time_messages = len([msg for msg in structured_messages if msg.get('type') == 'time'])
        my_messages = len([msg for msg in structured_messages if msg.get('type') == 'my_chat'])
        group_name_messages = len([msg for msg in structured_messages if msg.get('type') == 'group_name'])
        retract_messages = len([msg for msg in structured_messages if msg.get('type') == 'retract_message'])
        unknown_messages = len([msg for msg in structured_messages if msg.get('type') == 'unknown'])
        
        export_data["statistics"] = {
            "nickname_messages": nickname_messages,
            "time_messages": time_messages,
            "my_messages": my_messages,
            "group_name_messages": group_name_messages,
            "retract_messages": retract_messages,
            "unknown_messages": unknown_messages
        }
        
        # 保存到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"结构化聊天消息已导出到: {output_path}")
        print(f"  - 普通聊天消息: {nickname_messages} 条")
        print(f"  - 时间消息: {time_messages} 条")
        print(f"  - 我的消息: {my_messages} 条")
        print(f"  - 群聊名称: {group_name_messages} 条")
        print(f"  - 撤回消息: {retract_messages} 条")
        print(f"  - 未知内容: {unknown_messages} 条")
        
        return export_data
    
    def _organize_chat_messages(self, marked_texts):
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
    
    def _analyze_nicknames(self, marked_texts):
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
    
    def _is_group_name(self, nickname, position, nickname_analysis):
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
    
    def _is_time_content(self, text):
        """
        检查文本是否为时间内容
        
        Args:
            text: 要检查的文本
            
        Returns:
            是否为时间内容
        """
        import re
        
        # 时间模式列表（与_mark_time_content保持一致）
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


def main():
    """主函数"""
    # 初始化处理器
    processor = LongImageOCR(config_path="./default_rapidocr.yaml")
    
    # 处理长图
    # image_path = r"images/image copy 3.png"
    image_path = r"images/image copy 11.png"
    
    try:
        result = processor.process_long_image(image_path)
        print("\n处理结果摘要:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        
        # 导出汇总数据
        processor.export_summary_data()
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if os.path.exists("output_images"):
        shutil.rmtree("output_images")
    if os.path.exists("output_json"):
        shutil.rmtree("output_json")
    main()