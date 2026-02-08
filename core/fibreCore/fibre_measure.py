"""
A solution module based on widths between pair-edges of the fibre.
The core pipeline is: Bresenham iteration on normal direction + 8-domain detection + hardcode for minimum distance threshold

Author: waiwaiti
Date: 2026-02-08

"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes, medial_axis
from scipy.ndimage import gaussian_filter, sobel, distance_transform_edt
import warnings
warnings.filterwarnings('ignore')

def exclude_junction(skeleton_mask: np.ndarray, jer: float) -> np.ndarray:
    '''
    Exclude the part near intersection and crossing part of skeleton mask for more accurate measurement
    
    :param skeleton_mask:
    :type skeleton_mask: np.ndarray
    :param jer: Junction Exclusion Radius (JER), controls the distance from junction
    :type jer: float
    :return: The skeleton mask with the junction and intersection part removed
    :rtype: ndarray[Any, Any]
    '''
    padded = np.pad(skeleton_mask.astype(np.uint8), 1)
    neighbor_counts = np.zeros_like(skeleton_mask, dtype=np.int32)
    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for dy, dx in offsets:
        neighbor_counts += padded[1+dy:1+dy+skeleton_mask.shape[0],
                                  1+dx:1+dx+skeleton_mask.shape[1]]

    junction_mask = skeleton_mask & (neighbor_counts >= 3)

    # --- Junction exclusion zone ---
    if junction_mask.any():
        junc_map = np.zeros_like(skeleton_mask, dtype=np.uint8)
        junc_map[junction_mask] = 1
        dist_to_junc = distance_transform_edt(1 - junc_map)
        junction_exclusion_radius = jer 
        near_junction = dist_to_junc <= junction_exclusion_radius # type: ignore
    else:
        near_junction = np.zeros_like(skeleton_mask, dtype=bool)

    keep_mask = skeleton_mask & (neighbor_counts == 2) & (~near_junction)

    return keep_mask

def edge_width(ridge_mask: np.ndarray) -> float:
    '''
    Calulate the average width of ridge in mask. Needed to be combined with the internal fibre width for complete fibre diameter
    
    :param ridge_mask: 
    :type ridge_mask: np.ndarray
    :return: 
    :rtype: float
    '''
    centerline, distance_map = medial_axis(ridge_mask, return_distance=True)
    diameters = distance_map * 2
    refined_centerline = exclude_junction(centerline, 20)
    diameter_values = diameters[refined_centerline]
    return np.mean(diameter_values)

def bresenham_line(y0, x0, y1, x1) -> list:
    '''
    Returns all pixels along a line via Bresenham's algorithm
    
    :param y0: 
    :param x0: 
    :param y1: 
    :param x1: 
    :return: 
    :rtype: list
    '''
    points = []
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    x, y = x0, y0
    
    while True:
        points.append((y, x))
        
        if x == x1 and y == y1:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    
    return points


def measure_edge_pair_distances_final(edge_mask, 
                                       sample_rate=0.2,
                                       max_search_distance=50,
                                       min_distance_hard=5,
                                       smooth_sigma=1.0):
    """
    最终优化的边缘配对测量方法
    
    组合策略：
    1. Bresenham 遍历所有像素（不会跳过边缘）
    2. 前 min_distance_hard 个像素忽略（避免检测到自身）
    3. 之后使用8邻域检测（避免离散采样miss）
    
    参数：
    -------
    edge_mask : ndarray (bool)
        边缘线蒙版（True = 边缘，False = 其他）
    sample_rate : float (0, 1]
        采样率，控制计算量
        - 0.1 = 快速预览
        - 0.5 = 平衡
        - 1.0 = 完整测量
    max_search_distance : int
        沿法向最大搜索距离（像素）
        建议设为预期最大直径的1.5-2倍
    min_distance_hard : int
        硬编码最小距离（像素）
        前 N 个像素忽略，避免检测到自身边缘
        推荐：
        - 细边缘（1-2px宽）：3-5
        - 中等边缘（3-5px宽）：5-8
        - 粗边缘（>5px宽）：8-10
    smooth_sigma : float
        边缘平滑参数（用于计算法向）
        越大越平滑，但可能损失细节
    
    返回：
    -------
    edge_pairs : list of tuples
        [(coord1, coord2, distance), ...]
    distances : ndarray
        所有配对的距离数组
    measurements : dict
        统计信息
    """
    # 细化边缘到单像素宽
    skeleton = skeletonize(edge_mask)
    skeleton = exclude_junction(skeleton, 50)


    edge_coords = np.argwhere(skeleton)
    
    if len(edge_coords) == 0:
        return [], np.array([]), {'error': 'No edge points found'}
    
    # 计算法向方向
    edge_smooth = gaussian_filter(skeleton.astype(float), sigma=smooth_sigma)
    
    grad_y = sobel(edge_smooth, axis=0)
    grad_x = sobel(edge_smooth, axis=1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-8
    
    # 归一化法向
    normal_y = grad_y / grad_mag
    normal_x = grad_x / grad_mag
    
    # 采样
    n_total = len(edge_coords)
    n_sample = int(n_total * sample_rate)
    sample_indices = np.random.choice(n_total, min(n_sample, n_total), replace=False)
    
    # 辅助函数：检查8邻域是否有边缘
    def has_edge_in_8neighbors(y, x):
        h, w = skeleton.shape
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if skeleton[ny, nx]:
                        return True
        return False
    
    # 逐点搜索配对
    edge_pairs = []
    distances = []
    
    h, w = skeleton.shape
    
    for idx in sample_indices:
        y0, x0 = edge_coords[idx]
        
        # 法向
        ny, nx = normal_y[y0, x0], normal_x[y0, x0]
        
        # 法向太弱则跳过
        if abs(ny) < 0.1 and abs(nx) < 0.1:
            continue
        
        # 存储两个方向的候选配对
        candidates = []
        
        # 沿法向双向搜索
        for direction in [1, -1]:
            # 计算搜索终点
            end_y = int(y0 + direction * ny * max_search_distance)
            end_x = int(x0 + direction * nx * max_search_distance)
            
            # 边界裁剪
            end_y = max(0, min(h-1, end_y))
            end_x = max(0, min(w-1, end_x))
            
            # Bresenham 直线遍历
            line_points = bresenham_line(y0, x0, end_y, end_x)
            
            # 沿直线搜索边缘
            for i, (py, px) in enumerate(line_points):
                # 关键1：前 min_distance_hard 个像素忽略
                if i < min_distance_hard:
                    continue
                
                if i >= max_search_distance:
                    break
                
                # 关键2：之后使用8邻域检测
                if has_edge_in_8neighbors(py, px):
                    # 计算实际欧氏距离
                    dist = np.sqrt((py - y0)**2 + (px - x0)**2)
                    if dist > min_distance_hard * np.sqrt(2):
                        candidates.append(((py, px), dist))
                    break
        
        # 根据候选结果决定
        if len(candidates) == 0:
            # 两个方向都没找到，跳过
            continue
        elif len(candidates) == 1:
            # 只有一个方向找到了，使用该结果
            paired_coord, dist = candidates[0]
        else:
            # 两个方向都找到了，选择距离最小的
            paired_coord, dist = min(candidates, key=lambda c: c[1])
        
        edge_pairs.append(((y0, x0), paired_coord, dist))
        distances.append(dist)
    
    distances = np.array(distances)

    plt.figure()
    plt.imshow(skeleton, cmap='gray')
    show_n = min(1000, len(edge_pairs))
    for i in range(0, show_n, 3):
        (y1, x1), (y2, x2), dist = edge_pairs[i]
        color = plt.cm.jet(dist / 50.0) if dist < 50 else (1, 0, 0) # type: ignore
        plt.plot([x1, x2], [y1, y2], color=color, linewidth=1, alpha=0.6)
    plt.show()
    
    # 统计信息
    measurements = {
        'method': 'Bresenham + 8-Neighbor + Hard Min Distance',
        'n_pairs': len(distances),
        'n_sampled': n_sample,
        'sample_rate': sample_rate,
        'min_distance_hard': min_distance_hard,
        'mean_diameter': distances.mean() if len(distances) > 0 else np.nan,
        'std_diameter': distances.std() if len(distances) > 0 else np.nan,
        'median_diameter': np.median(distances) if len(distances) > 0 else np.nan,
        'min_diameter': distances.min() if len(distances) > 0 else np.nan,
        'max_diameter': distances.max() if len(distances) > 0 else np.nan,
        'percentiles': {
            p: np.percentile(distances, p) if len(distances) > 0 else np.nan
            for p in [10, 25, 50, 75, 90, 95]
        }
    }
    
    return edge_pairs, distances, measurements


def visualize_results(edge_mask, pairs, distances, stats, save_path='final_results.png'):
    """
    可视化测量结果
    
    参数：
    -------
    edge_mask : ndarray
        原始边缘蒙版
    pairs : list
        配对列表
    distances : ndarray
        距离数组
    stats : dict
        统计信息
    save_path : str
        保存路径
    """
    import matplotlib.pyplot as plt
    from skimage.morphology import skeletonize
    
    skeleton = skeletonize(edge_mask)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # 1. 所有配对
    axes[0, 0].imshow(skeleton, cmap='gray')
    show_n = min(1000, len(pairs))
    for i in range(0, show_n, 3):
        (y1, x1), (y2, x2), dist = pairs[i]
        color = plt.cm.jet(dist / 50.0) if dist < 50 else (1, 0, 0) # type: ignore
        axes[0, 0].plot([x1, x2], [y1, y2], color=color, linewidth=0.5, alpha=0.6)
    
    axes[0, 0].set_title(f'配对可视化\n{len(pairs):,} 对，颜色=距离', 
                        fontsize=13, weight='bold')
    axes[0, 0].axis('off')
    
    # 2. 局部放大
    crop_y, crop_x, crop_size = 300, 500, 200
    axes[0, 1].imshow(skeleton[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size], cmap='gray')
    
    for (y1, x1), (y2, x2), dist in pairs:
        if crop_y <= y1 < crop_y+crop_size and crop_x <= x1 < crop_x+crop_size:
            axes[0, 1].plot([x1-crop_x, x2-crop_x], [y1-crop_y, y2-crop_y], 
                           'r-', linewidth=1.5, alpha=0.7)
            axes[0, 1].plot(x1-crop_x, y1-crop_y, 'go', markersize=4)
            axes[0, 1].plot(x2-crop_x, y2-crop_y, 'bo', markersize=4)
    
    axes[0, 1].set_title('局部放大\n绿→蓝配对', fontsize=13)
    axes[0, 1].axis('off')
    
    # 3. 距离分布
    axes[1, 0].hist(distances, bins=40, color='blue', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(np.median(distances), color='red', linestyle='--', 
                      linewidth=2, label=f'中位数: {np.median(distances):.1f}px')
    axes[1, 0].axvline(np.mean(distances), color='green', linestyle='--', 
                      linewidth=2, label=f'平均: {np.mean(distances):.1f}px')
    axes[1, 0].set_xlabel('距离 (pixels)', fontsize=12)
    axes[1, 0].set_ylabel('频数', fontsize=12)
    axes[1, 0].set_title('距离分布直方图', fontsize=13)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(alpha=0.3)
    
    # 4. 统计信息
    axes[1, 1].axis('off')
    stats_text = f"""
测量统计

方法: {stats['method']}

配对数: {stats['n_pairs']:,}
采样率: {stats['sample_rate']*100:.0f}%
硬编码最小距离: {stats['min_distance_hard']} px

直径统计:
  平均: {stats['mean_diameter']:.2f} ± {stats['std_diameter']:.2f} px
  中位数: {stats['median_diameter']:.2f} px
  范围: [{stats['min_diameter']:.0f}, {stats['max_diameter']:.0f}] px

分位数:
  10%: {stats['percentiles'][10]:.2f} px
  25%: {stats['percentiles'][25]:.2f} px
  50%: {stats['percentiles'][50]:.2f} px
  75%: {stats['percentiles'][75]:.2f} px
  90%: {stats['percentiles'][90]:.2f} px
  95%: {stats['percentiles'][95]:.2f} px
    """
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    # plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


# === 使用示例 ===
if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    
    # 读取边缘蒙版
    edge_mask = np.array(Image.open(r"E:\CoraMetix\Fibre Diameter Measurement\scaffoldAnalysis_Dev\mask.jpg").convert('L')) > 128
    print(edge_width(edge_mask))
    
    print("=== 边缘配对纤维测宽 - 最终版本 ===\n")
    
    # 测量
    pairs, distances, stats = measure_edge_pair_distances_final(
        edge_mask,
        sample_rate=0.2,           # 20%采样
        max_search_distance=50,    # 最大搜索50像素
        min_distance_hard=5,       # 前5个像素忽略
        smooth_sigma=1.0
    )
    
    print(f"配对数: {stats['n_pairs']:,}")
    print(f"\n直径统计:")
    print(f"  平均: {stats['mean_diameter']:.2f} ± {stats['std_diameter']:.2f} px")
    print(f"  中位数: {stats['median_diameter']:.2f} px")
    print(f"  范围: [{stats['min_diameter']:.0f}, {stats['max_diameter']:.0f}] px")
    
    print(f"\n分位数:")
    for p in [10, 25, 50, 75, 90, 95]:
        print(f"  {p:2d}%: {stats['percentiles'][p]:.2f} px")
    
    # 可视化
    visualize_results(edge_mask, pairs, distances, stats, 'final_edge_pairing_results.png')
    
    # print("\n✓ 结果保存: final_edge_pairing_results.png")


# === 参数调优指南 ===
"""
根据你的数据调整参数：

1. min_distance_hard（硬编码最小距离）:
   - 观察你的边缘线宽度
   - 如果边缘是1-2像素宽 → 设为 3-5
   - 如果边缘是3-5像素宽 → 设为 5-8
   - 如果边缘是>5像素宽 → 设为 8-10
   
   判断标准：
   - 太小：会检测到自身边缘 → 距离过小（中位数<5）
   - 太大：会漏掉细纤维 → 配对数下降
   
2. max_search_distance:
   - 设为预期最粗纤维直径的1.5-2倍
   - 如果最粗纤维~30px → 设为 50-60
   
3. sample_rate:
   - 0.1 = 快速预览
   - 0.5 = 平衡（推荐）
   - 1.0 = 完整测量（慢但准确）

调试流程：
1. 先用 sample_rate=0.1 快速测试
2. 调整 min_distance_hard，观察中位数
3. 确认参数后，用 sample_rate=0.5 或 1.0 完整测量
"""
