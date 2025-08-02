import cv2
import numpy as np


def preprocess_image(image_path):
    """
    预处理图像：读取、转灰度、模糊、二值化
    
    Args:
        image_path (str): 图像文件路径
        
    Returns:
        tuple: (原图, 二值化图像)
    """
    # 读取图像
    if isinstance(image_path, str): 
        img = cv2.imread(image_path)
    else:
        img = image_path
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 二值化（使用 Otsu 自动阈值）
    # _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY)

    
    # 反转颜色（头像通常是深色背景）
    binary = 255 - binary
    
    return img, binary


def extract_contours_and_rects(binary_img, img):
    """
    提取轮廓并计算外接矩形
    
    Args:
        binary_img: 二值化图像
        
    Returns:
        list: 按面积排序的外接矩形列表 [(x, y, w, h), ...]
    """
    # 查找轮廓
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 计算所有轮廓的外接矩形
    rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rects.append((x, y, w, h))
    
    #
    # 按面积从大到小排序
    rects = sorted(rects, key=lambda box: box[2] * box[3], reverse=True)
    # 将这些外接矩形画到原图中（用于调试或可视化）
    # 注意：此处假设原图变量名为 img，如果没有可传参或略作修改
    # 这里只做演示，实际调用时请确保 img 已定义
    # import cv2  # 如果未导入请在文件头部导入
    result_img = img.copy()
    for (x, y, w, h) in rects:
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imwrite("rects_visualization.jpg", result_img)
    pass

    # 找到最左侧的前四个近似正方形的框，并画出来
    if rects:
        # 计算“几乎为正方形”的阈值，这里设为宽高比在0.9~1.1之间
        square_rects = [r for r in rects if 0.9 <= r[2]/r[3] <= 1.1 or 0.9 <= r[3]/r[2] <= 1.1]
        if square_rects:
            # 按x坐标排序，取最左侧的前四个
            leftmost_three = sorted(square_rects, key=lambda r: r[0])[:4]
            
            # 取面积最大的那个box的x_croped
            max_area_box = max(square_rects, key=lambda r: r[2]*r[3])
            x_croped = max_area_box[0] + max_area_box[2] + 3
            img_square = img.copy()
            for idx, rect in enumerate(leftmost_three):
                x, y, w, h = rect
                # 用不同颜色区分，或者都用红色
                cv2.rectangle(img_square, (x, y), (x + w, y + h), (0, 0, 255), 3)
                print(f"第{idx+1}个最左侧且近似正方形的框: {rect}")
            # cv2.imwrite("leftmost_three_square_rects.jpg", img_square)
            
            print("最左侧的前三个近似正方形的框已画出，保存为 leftmost_three_square_rects.jpg")
        else:
            print("未找到近似正方形的框")
    else:
        print("没有检测到任何外接矩形")

    
    return rects,x_croped



def extract_contours_and_rects_croped(binary_img, img):
    """
    提取轮廓并计算外接矩形
    
    Args:
        binary_img: 二值化图像
        
    Returns:
        list: 按面积排序的外接矩形列表 [(x, y, w, h), ...]
    """
    # 查找轮廓
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 计算所有轮廓的外接矩形
    rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rects.append((x, y, w, h))
    
    # 按面积从大到小排序
    rects = sorted(rects, key=lambda box: box[2] * box[3], reverse=True)
    # 在原图上绘制所有轮廓的外接矩形，并保存
    img_all_rects = img.copy()
    for rect in rects:
        x, y, w, h = rect
        cv2.rectangle(img_all_rects, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imwrite("all_rects_on_croped.jpg", img_all_rects)
    print("所有轮廓的外接矩形已画出，保存为 all_rects_on_croped.jpg")

    # 保留几乎为正方形的框（宽高比在0.9~1.1之间）
    square_rects = [r for r in rects if 0.9 <= r[2]/r[3] <= 1.1 or 0.9 <= r[3]/r[2] <= 1.1]

    if square_rects:
        # 统计所有正方形框的w和h
        ws = [r[2] for r in square_rects]
        hs = [r[3] for r in square_rects]
        mean_w = sum(ws) / len(ws)
        mean_h = sum(hs) / len(hs)
        std_w = (sum([(w - mean_w) ** 2 for w in ws]) / len(ws)) ** 0.5
        std_h = (sum([(h - mean_h) ** 2 for h in hs]) / len(hs)) ** 0.5

        # 只保留w和h都接近均值的正方形框（这里用1个标准差为阈值，可根据实际调整）
        filtered_square_rects = [
            r for r in square_rects
            if abs(r[2] - mean_w) <= std_w and abs(r[3] - mean_h) <= std_h
        ]

        if filtered_square_rects:
            # 按x坐标排序，取最左侧的前四个
            leftmost_four = sorted(filtered_square_rects, key=lambda r: r[0])[:]

            # 取面积最大的那个box的x_croped
            max_area_box = max(filtered_square_rects, key=lambda r: r[2]*r[3])
            x_croped = max_area_box[0] + max_area_box[2] + 3

            img_square = img.copy()
            for idx, rect in enumerate(leftmost_four):
                x, y, w, h = rect
                cv2.rectangle(img_square, (x, y), (x + w, y + h), (0, 0, 255), 3)
                print(f"第{idx+1}个最左侧且尺寸接近的正方形框: {rect}")
            # cv2.imwrite("leftmost_four_square_rects_filtered.jpg", img_square)
            print("最左侧的前四个尺寸接近的正方形框已画出，保存为 leftmost_four_square_rects_filtered.jpg")
        else:
            print("未找到尺寸接近的正方形框")
            x_croped = None
    else:
        print("未找到近似正方形的框")
        x_croped = None

    return filtered_square_rects if 'filtered_square_rects' in locals() else [], x_croped


def calculate_iou(box1, box2):
    """
    计算两个框的IOU（交并比）
    
    Args:
        box1, box2: 格式为 (x, y, w, h) 的边界框
        
    Returns:
        float: IOU值
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # 计算交集
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    
    # 计算并集
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
    
    return inter_area / union_area


def apply_nms(rects, iou_threshold=0.0):
    """
    应用非最大抑制（NMS）算法去除重叠框
    
    Args:
        rects: 边界框列表
        iou_threshold: IOU阈值
        
    Returns:
        list: 经过NMS筛选的边界框列表
    """
    keep_rects = []
    for rect in rects:
        keep = True
        for kept_rect in keep_rects:
            if calculate_iou(rect, kept_rect) > iou_threshold:
                keep = False
                break
        if keep:
            keep_rects.append(rect)
    
    return keep_rects


def should_merge(box1, box2, distance_threshold):
    """
    判断两个框是否应该合并（基于距离和IOU）
    
    Args:
        box1, box2: 边界框
        distance_threshold: 距离阈值
        
    Returns:
        bool: 是否应该合并
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # 计算两个框中心点之间的距离
    center1_x, center1_y = x1 + w1 // 2, y1 + h1 // 2
    center2_x, center2_y = x2 + w2 // 2, y2 + h2 // 2
    distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
    
    # 如果距离小于阈值或IOU大于0.01，则合并
    return distance < distance_threshold or calculate_iou(box1, box2) > 0.01


def merge_boxes(boxes):
    """
    合并一组框为一个最小外接框
    
    Args:
        boxes: 边界框列表
        
    Returns:
        tuple: 合并后的边界框 (x, y, w, h)
    """
    if not boxes:
        return None
    
    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[0] + box[2] for box in boxes)
    max_y = max(box[1] + box[3] for box in boxes)
    
    return (min_x, min_y, max_x - min_x, max_y - min_y)


def merge_nearby_boxes(rects, merge_threshold):
    """
    合并相邻的边界框
    
    Args:
        rects: 边界框列表
        merge_threshold: 合并距离阈值
        
    Returns:
        list: 合并后的边界框列表
    """
    merged_rects = []
    used = [False] * len(rects)
    
    for i in range(len(rects)):
        if used[i]:
            continue
        
        # 找到所有需要与当前框合并的框
        group = [rects[i]]
        used[i] = True
        
        for j in range(i + 1, len(rects)):
            if used[j]:
                continue
            
            # 检查是否与组中任何一个框相邻
            should_add = False
            for box_in_group in group:
                if should_merge(box_in_group, rects[j], merge_threshold):
                    should_add = True
                    break
            
            if should_add:
                group.append(rects[j])
                used[j] = True
        
        # 合并这一组框
        merged_box = merge_boxes(group)
        if merged_box:
            merged_rects.append(merged_box)
    
    return merged_rects


def calculate_merge_threshold(rects):
    """
    根据最大框计算合并阈值
    
    Args:
        rects: 边界框列表
        
    Returns:
        int: 合并阈值
    """
    if rects:
        max_box = rects[0]  # 面积最大的框
        max_dim = max(max_box[2], max_box[3])  # 最长边
        merge_threshold = max_dim
        print(f"合并阈值: {merge_threshold}像素（面积最大框的最长边）")
        return merge_threshold
    else:
        print("未找到框，使用默认合并阈值: 20像素")
        return 20


def draw_rectangles(img, rects, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制矩形框
    
    Args:
        img: 输入图像
        rects: 边界框列表
        color: 矩形颜色 (B, G, R)
        thickness: 线条粗细
        
    Returns:
        np.ndarray: 绘制了矩形的图像副本
    """
    result_img = img.copy()
    for (x, y, w, h) in rects:
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, thickness)
    return result_img


def main():
    """主函数：执行完整的头像检测流程"""
    try:
        # 图像预处理
        img, binary = preprocess_image("test_images\images_5.jpg")
        
        # 保存中间结果
        # cv2.imwrite("binary_ori.png", 255 - binary)  # 保存原始二值化图像
        # cv2.imwrite("binary_reversed_ori.png", binary)  # 保存反转后的二值化图像

        # 对binary进行闭运算（先膨胀后腐蚀，填补小黑洞）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        # cv2.imwrite("binary_closed_ori.png", binary)
        
        # 提取轮廓和外接矩形
        rects,x_croped = extract_contours_and_rects(binary, img)
        print(f"检测到 {len(rects)} 个候选区域")

        croped_image = img[0:img.shape[0],0:x_croped]
        # cv2.imwrite("croped_image_ori.jpg", croped_image)


        # 对croped_image进行预处理
        img, binary = preprocess_image(croped_image)
        
        # 保存中间结果
        # cv2.imwrite("binary.png", 255 - binary)  # 保存原始二值化图像
        # cv2.imwrite("binary_reversed.png", binary)  # 保存反转后的二值化图像

        # 对binary进行闭运算（先膨胀后腐蚀，填补小黑洞）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        # cv2.imwrite("binary_closed.png", binary)
        
        # 提取轮廓和外接矩形
        rects,x_croped = extract_contours_and_rects(binary, img)
        print(f"检测到 {len(rects)} 个候选区域")


        
        # 计算合并阈值
        merge_threshold = calculate_merge_threshold(rects)
        
        # 应用NMS去除重叠框
        nms_rects = apply_nms(rects, iou_threshold=0.0)
        print(f"NMS后保留 {len(nms_rects)} 个区域")
        
        # 合并相邻的框
        merged_rects = merge_nearby_boxes(nms_rects, merge_threshold)
        # INSERT_YOUR_CODE
        # 将合并后的框画出来并保存
        # merged_rects_img = img.copy()
        # for (x, y, w, h) in merged_rects:
        #     cv2.rectangle(merged_rects_img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
        # cv2.imwrite("merged_rects_drawn.jpg", merged_rects_img)
        # print("合并后的框已画出，保存为 merged_rects_drawn.jpg")
        
        
        # 保存结果图像
        # nms_result = draw_rectangles(img, nms_rects, color=(0, 255, 0))
        # cv2.imwrite("nms_result.jpg", nms_result)
        
        merged_result = draw_rectangles(img, merged_rects, color=(255, 0, 0))
        cv2.imwrite("merged_result.jpg", merged_result)
        
        # # 绘制所有原始轮廓（用于对比）
        # all_contours_result = draw_rectangles(img, rects, color=(0, 255, 0))
        # cv2.imwrite("result_with_binary.jpg", all_contours_result)
        
        # print("处理完成！结果已保存到以下文件：")
        # print("- binary.png: 原始二值化图像")
        # print("- binary_reversed.png: 反转后的二值化图像")
        # print("- nms_result.jpg: NMS处理结果")
        print("- merged_result.jpg: 最终合并结果")
        # print("- result_with_binary.jpg: 所有原始轮廓")
        return merged_result
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")


if __name__ == "__main__":
    main()




