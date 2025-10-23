import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import math

def load_image_color(image_path):
    """載入彩色圖像"""
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img, dtype=np.float64)
    return img_array

def svd_compress_color(img_array, k):
    """對彩色圖像的三個通道分別進行SVD壓縮"""
    R_compressed = svd_compress_channel(img_array[:,:,0], k)
    G_compressed = svd_compress_channel(img_array[:,:,1], k)
    B_compressed = svd_compress_channel(img_array[:,:,2], k)
    
    compressed_color = np.stack([R_compressed, G_compressed, B_compressed], axis=2)
    return compressed_color

def svd_compress_channel(channel, k):
    """對單個通道使用SVD壓縮"""
    U, s, Vt = np.linalg.svd(channel, full_matrices=False)
    compressed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    return compressed

def calculate_energy_retention(s, k):
    """計算能量保留比例"""
    return np.sum(s[:k]**2) / np.sum(s**2)

def mse(a, ak):
    """計算均方誤差"""
    return np.mean((a.astype(np.float64) - ak.astype(np.float64)) ** 2)

def calculate_psnr(original, compressed):
    """計算PSNR值"""
    mse_val = mse(original, compressed)
    if mse_val < 1.0e-10:
        return 100.0
    return 10 * math.log10(255.0**2 / mse_val)

def cr(originalSize, sampleSize):
    """計算壓縮率 (CR = 原始大小 / 壓縮後大小)"""
    if sampleSize == 0:
        return float('inf')
    return originalSize / sampleSize

def find_optimal_k_energy(s_list, threshold=0.95):
    """基於能量保留比例找最佳K值（對三個通道取平均）"""
    k_values = []
    energy_ratios = []
    
    for s in s_list:
        total_energy = np.sum(s**2)
        cumulative_energy = np.cumsum(s**2)
        energy_ratio = cumulative_energy / total_energy
        optimal_k = np.argmax(energy_ratio >= threshold) + 1
        k_values.append(optimal_k)
        energy_ratios.append(energy_ratio)
    
    # 取三個通道的平均K值
    avg_k = int(np.mean(k_values))
    return avg_k, energy_ratios

def find_optimal_k_elbow(s_list):
    """使用肘部法則找最佳K值（對三個通道取平均）"""
    k_values = []
    
    for s in s_list:
        s_normalized = s / s[0]
        if len(s) > 2:
            second_diff = np.diff(np.diff(s_normalized))
            optimal_k = np.argmax(np.abs(second_diff)) + 2
        else:
            optimal_k = len(s) // 2
        k_values.append(optimal_k)
    
    # 取三個通道的平均K值
    avg_k = int(np.mean(k_values))
    return avg_k

def find_optimal_k_psnr(img_array, s_list, U_list, Vt_list, target_psnr=30):
    """基於目標PSNR找最佳K值"""
    max_k = min(len(s) for s in s_list)
    
    for k in range(1, max_k + 1):
        compressed = svd_compress_color(img_array, k)
        psnr = calculate_psnr(img_array, compressed)
        if psnr >= target_psnr:
            return k, psnr
    
    # 如果達不到目標PSNR，返回最大K值
    compressed = svd_compress_color(img_array, max_k)
    return max_k, calculate_psnr(img_array, compressed)

def save_compressed_image_color(img_array, k, output_path):
    """保存壓縮後的彩色圖像並返回檔案大小"""
    compressed = svd_compress_color(img_array, k)
    compressed_uint8 = np.clip(compressed, 0, 255).astype(np.uint8)
    Image.fromarray(compressed_uint8).save(output_path)
    file_size_kb = round(os.stat(output_path).st_size / 1024, 2)
    return file_size_kb

def visualize_results_color(img_array, s_list, energy_ratios, k_energy_95, k_elbow, k_psnr_30, 
                            original_size_kb, output_dir='./image'):
    """視覺化分析結果（彩色版本）"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 14))
    
    # 第一行：4張圖像
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(img_array.astype(np.uint8))
    ax1.set_title('Original Image', fontsize=11, fontweight='bold')
    ax1.axis('off')
    
    # 95% Energy
    ax2 = plt.subplot(3, 4, 2)
    compressed1 = svd_compress_color(img_array, k_energy_95)
    psnr1 = calculate_psnr(img_array, compressed1)
    ax2.imshow(np.clip(compressed1, 0, 255).astype(np.uint8))
    ax2.set_title(f'95% Energy\nk={k_energy_95}\nPSNR={psnr1:.2f}dB', fontsize=10)
    ax2.axis('off')
    
    # Elbow Method
    ax3 = plt.subplot(3, 4, 3)
    compressed2 = svd_compress_color(img_array, k_elbow)
    psnr2 = calculate_psnr(img_array, compressed2)
    ax3.imshow(np.clip(compressed2, 0, 255).astype(np.uint8))
    ax3.set_title(f'Elbow Method\nk={k_elbow}\nPSNR={psnr2:.2f}dB', fontsize=10)
    ax3.axis('off')
    
    # PSNR>=30dB
    ax4 = plt.subplot(3, 4, 4)
    compressed3 = svd_compress_color(img_array, k_psnr_30)
    psnr3 = calculate_psnr(img_array, compressed3)
    ax4.imshow(np.clip(compressed3, 0, 255).astype(np.uint8))
    ax4.set_title(f'PSNR>=30dB\nk={k_psnr_30}\nPSNR={psnr3:.2f}dB', fontsize=10)
    ax4.axis('off')
    
    # 第二行：奇異值衰減曲線（三個通道）
    ax5 = plt.subplot(3, 4, (5, 6))
    colors_channel = ['red', 'green', 'blue']
    channel_names = ['R Channel', 'G Channel', 'B Channel']
    
    for i, (s, color, name) in enumerate(zip(s_list, colors_channel, channel_names)):
        ax5.plot(s, color=color, linewidth=1.5, alpha=0.7, label=name)
    
    ax5.axvline(x=k_energy_95, color='darkgreen', linestyle='--', linewidth=2, 
                label=f'95% Energy (k={k_energy_95})')
    ax5.axvline(x=k_elbow, color='orange', linestyle='--', linewidth=2, 
                label=f'Elbow (k={k_elbow})')
    ax5.axvline(x=k_psnr_30, color='purple', linestyle='--', linewidth=2, 
                label=f'PSNR (k={k_psnr_30})')
    
    ax5.set_xlabel('Singular Value Index', fontsize=10)
    ax5.set_ylabel('Singular Value Magnitude', fontsize=10)
    ax5.set_title('Singular Value Decay Curve (RGB Channels)', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=7, loc='upper right')
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    
    # 能量累積曲線（三個通道的平均）
    ax6 = plt.subplot(3, 4, (7, 8))
    avg_energy_ratio = np.mean(energy_ratios, axis=0)
    
    for i, (energy_ratio, color, name) in enumerate(zip(energy_ratios, colors_channel, channel_names)):
        ax6.plot(energy_ratio, color=color, linewidth=1, alpha=0.5)
    
    ax6.plot(avg_energy_ratio, 'k-', linewidth=2, label='Average')
    ax6.axhline(y=0.90, color='b', linestyle=':', linewidth=1, label='90% Threshold')
    ax6.axhline(y=0.95, color='g', linestyle='--', linewidth=1.5, label='95% Threshold')
    ax6.axhline(y=0.99, color='purple', linestyle=':', linewidth=1, label='99% Threshold')
    ax6.axvline(x=k_energy_95, color='g', linestyle='--', linewidth=1.5, alpha=0.5)
    
    ax6.set_xlabel('K Value', fontsize=10)
    ax6.set_ylabel('Cumulative Energy Ratio', fontsize=10)
    ax6.set_title('Energy Accumulation Curve (RGB Channels)', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 1.05])
    
    # 第三行：PSNR vs K值曲線
    ax7 = plt.subplot(3, 4, (9, 10))
    k_range = range(1, min(200, min(len(s) for s in s_list)), 3)
    psnr_values = [calculate_psnr(img_array, svd_compress_color(img_array, k)) for k in k_range]
    
    ax7.plot(list(k_range), psnr_values, 'g-', linewidth=2)
    ax7.axhline(y=30, color='orange', linestyle='--', linewidth=1.5, label='30dB Threshold')
    ax7.axhline(y=35, color='r', linestyle='--', linewidth=1.5, label='35dB Threshold')
    ax7.axvline(x=k_psnr_30, color='r', linestyle='--', linewidth=1.5, alpha=0.5)
    
    ax7.set_xlabel('K Value', fontsize=10)
    ax7.set_ylabel('PSNR (dB)', fontsize=10)
    ax7.set_title('PSNR vs K Value Curve (Color Image)', fontsize=11, fontweight='bold')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)
    
    # 壓縮率比較
    ax8 = plt.subplot(3, 4, (11, 12))
    methods = ['95% Energy', 'Elbow Method', 'PSNR>=30dB']
    k_values = [k_energy_95, k_elbow, k_psnr_30]
    
    compression_ratios = []
    for i, k in enumerate(k_values):
        temp_path = os.path.join(output_dir, f'temp_compressed_color_{k}.jpg')
        compressed_size = save_compressed_image_color(img_array, k, temp_path)
        cr_value = cr(original_size_kb, compressed_size)
        compression_ratios.append(cr_value)
    
    colors = ['green', 'orange', 'red']
    bars = ax8.bar(methods, compression_ratios, color=colors, alpha=0.7, edgecolor='black')
    ax8.set_ylabel('Compression Ratio (CR)', fontsize=10)
    ax8.set_title('Compression Ratio Comparison (Color Image)', fontsize=11, fontweight='bold')
    ax8.grid(True, axis='y', alpha=0.3)
    
    for bar, k_val, ratio in zip(bars, k_values, compression_ratios):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'k={k_val}\nCR={ratio:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('svd_analysis_results_color.png', dpi=150, bbox_inches='tight')
    print("\n視覺化結果已保存為 'svd_analysis_results_color.png'")
    plt.show()

def analyze_svd_compression_color(image_path, output_dir='./image'):
    """完整的SVD壓縮分析（彩色版本）"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 載入彩色圖像
    img_array = load_image_color(image_path)
    print(f"圖像尺寸: {img_array.shape}")
    
    # 保存原始彩色圖並獲取大小
    original_path = os.path.join(output_dir, 'original_color.jpg')
    Image.fromarray(img_array.astype(np.uint8)).save(original_path)
    original_size_kb = round(os.stat(original_path).st_size / 1024, 2)
    print(f"原始圖像大小: {original_size_kb} KB")
    
    # 對三個通道分別執行SVD分解
    U_list = []
    s_list = []
    Vt_list = []
    
    print("\n對 RGB 三個通道分別進行 SVD 分解...")
    for i, channel_name in enumerate(['R', 'G', 'B']):
        U, s, Vt = np.linalg.svd(img_array[:,:,i], full_matrices=False)
        U_list.append(U)
        s_list.append(s)
        Vt_list.append(Vt)
        print(f"{channel_name} 通道奇異值數量: {len(s)}")
    
    # 能量保留法
    k_energy_90, energy_ratios = find_optimal_k_energy(s_list, threshold=0.90)
    k_energy_95, _ = find_optimal_k_energy(s_list, threshold=0.95)
    k_energy_99, _ = find_optimal_k_energy(s_list, threshold=0.99)
    
    # 計算真實的壓縮率
    size_90 = save_compressed_image_color(img_array, k_energy_90, 
                                          os.path.join(output_dir, f'compressed_color_{k_energy_90}.jpg'))
    size_95 = save_compressed_image_color(img_array, k_energy_95, 
                                          os.path.join(output_dir, f'compressed_color_{k_energy_95}.jpg'))
    size_99 = save_compressed_image_color(img_array, k_energy_99, 
                                          os.path.join(output_dir, f'compressed_color_{k_energy_99}.jpg'))
    
    cr_90 = cr(original_size_kb, size_90)
    cr_95 = cr(original_size_kb, size_95)
    cr_99 = cr(original_size_kb, size_99)
    
    # 計算PSNR
    psnr_90 = calculate_psnr(img_array, svd_compress_color(img_array, k_energy_90))
    psnr_95 = calculate_psnr(img_array, svd_compress_color(img_array, k_energy_95))
    psnr_99 = calculate_psnr(img_array, svd_compress_color(img_array, k_energy_99))
    
    print(f"\n方法1 - 能量保留法:")
    print(f"  90%能量: k = {k_energy_90}, 檔案大小 = {size_90}KB, CR = {cr_90:.2f}, PSNR = {psnr_90:.2f}dB")
    print(f"  95%能量: k = {k_energy_95}, 檔案大小 = {size_95}KB, CR = {cr_95:.2f}, PSNR = {psnr_95:.2f}dB")
    print(f"  99%能量: k = {k_energy_99}, 檔案大小 = {size_99}KB, CR = {cr_99:.2f}, PSNR = {psnr_99:.2f}dB")
    
    # 肘部法則
    k_elbow = find_optimal_k_elbow(s_list)
    size_elbow = save_compressed_image_color(img_array, k_elbow, 
                                             os.path.join(output_dir, f'compressed_color_{k_elbow}.jpg'))
    cr_elbow = cr(original_size_kb, size_elbow)
    psnr_elbow = calculate_psnr(img_array, svd_compress_color(img_array, k_elbow))
    
    print(f"\n方法2 - 肘部法則:")
    print(f"  最佳k = {k_elbow}, 檔案大小 = {size_elbow}KB, CR = {cr_elbow:.2f}, PSNR = {psnr_elbow:.2f}dB")
    
    # PSNR閾值法
    k_psnr_30, psnr_30 = find_optimal_k_psnr(img_array, s_list, U_list, Vt_list, target_psnr=30)
    k_psnr_35, psnr_35 = find_optimal_k_psnr(img_array, s_list, U_list, Vt_list, target_psnr=35)
    
    size_psnr_30 = save_compressed_image_color(img_array, k_psnr_30, 
                                               os.path.join(output_dir, f'compressed_color_{k_psnr_30}.jpg'))
    size_psnr_35 = save_compressed_image_color(img_array, k_psnr_35, 
                                               os.path.join(output_dir, f'compressed_color_{k_psnr_35}.jpg'))
    
    cr_psnr_30 = cr(original_size_kb, size_psnr_30)
    cr_psnr_35 = cr(original_size_kb, size_psnr_35)
    
    print(f"\n方法3 - PSNR閾值法:")
    print(f"  PSNR>=30dB: k = {k_psnr_30}, 實際PSNR = {psnr_30:.2f}dB, 檔案大小 = {size_psnr_30}KB, CR = {cr_psnr_30:.2f}")
    print(f"  PSNR>=35dB: k = {k_psnr_35}, 實際PSNR = {psnr_35:.2f}dB, 檔案大小 = {size_psnr_35}KB, CR = {cr_psnr_35:.2f}")
    
    # 視覺化結果
    visualize_results_color(img_array, s_list, energy_ratios, k_energy_95, k_elbow, k_psnr_30, 
                           original_size_kb, output_dir)
    
    return {
        'k_energy_90': k_energy_90,
        'k_energy_95': k_energy_95,
        'k_energy_99': k_energy_99,
        'k_elbow': k_elbow,
        'k_psnr_30': k_psnr_30,
        'k_psnr_35': k_psnr_35,
        'cr_90': cr_90,
        'cr_95': cr_95,
        'cr_99': cr_99,
        'cr_elbow': cr_elbow,
        'cr_psnr_30': cr_psnr_30,
        'cr_psnr_35': cr_psnr_35,
        'psnr_90': psnr_90,
        'psnr_95': psnr_95,
        'psnr_99': psnr_99,
        'psnr_elbow': psnr_elbow,
        'psnr_30': psnr_30,
        'psnr_35': psnr_35
    }

if __name__ == "__main__":
    image_path = 'matrixaimg.jpg'
    
    results = analyze_svd_compression_color(image_path)
    
    print("\n" + "="*70)
    print("建議的K值總結 (彩色圖像):")
    print("="*70)
    print(f"高壓縮率 (90%能量): k = {results['k_energy_90']}, CR = {results['cr_90']:.2f}, PSNR = {results['psnr_90']:.2f}dB")
    print(f"平衡品質 (95%能量): k = {results['k_energy_95']}, CR = {results['cr_95']:.2f}, PSNR = {results['psnr_95']:.2f}dB (推薦)")
    print(f"高品質 (99%能量): k = {results['k_energy_99']}, CR = {results['cr_99']:.2f}, PSNR = {results['psnr_99']:.2f}dB")
    print(f"肘部法則建議: k = {results['k_elbow']}, CR = {results['cr_elbow']:.2f}, PSNR = {results['psnr_elbow']:.2f}dB")
    print(f"PSNR>=30dB: k = {results['k_psnr_30']}, CR = {results['cr_psnr_30']:.2f}, PSNR = {results['psnr_30']:.2f}dB")
    print(f"PSNR>=35dB: k = {results['k_psnr_35']}, CR = {results['cr_psnr_35']:.2f}, PSNR = {results['psnr_35']:.2f}dB")
    print("="*70)
