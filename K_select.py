import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import math

def load_image(image_path):
    """載入圖像並轉換為灰階"""
    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype=np.float64)
    return img_array

def svd_compress(img_array, k):
    """使用SVD壓縮圖像，保留前k個奇異值"""
    U, s, Vt = np.linalg.svd(img_array, full_matrices=False)
    img_compressed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    return img_compressed

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

def find_optimal_k_energy(s, threshold=0.95):
    """基於能量保留比例找最佳K值"""
    total_energy = np.sum(s**2)
    cumulative_energy = np.cumsum(s**2)
    energy_ratio = cumulative_energy / total_energy
    optimal_k = np.argmax(energy_ratio >= threshold) + 1
    return optimal_k, energy_ratio

def find_optimal_k_elbow(s):
    """使用肘部法則找最佳K值"""
    s_normalized = s / s[0]
    if len(s) > 2:
        second_diff = np.diff(np.diff(s_normalized))
        optimal_k = np.argmax(np.abs(second_diff)) + 2
    else:
        optimal_k = len(s) // 2
    return optimal_k

def find_optimal_k_psnr(img_array, s, target_psnr=30):
    """基於目標PSNR找最佳K值"""
    U, _, Vt = np.linalg.svd(img_array, full_matrices=False)
    
    for k in range(1, len(s) + 1):
        compressed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        psnr = calculate_psnr(img_array, compressed)
        if psnr >= target_psnr:
            return k, psnr
    
    return len(s), calculate_psnr(img_array, U @ np.diag(s) @ Vt)

def save_compressed_image(img_array, k, output_path):
    """保存壓縮後的圖像並返回檔案大小"""
    compressed = svd_compress(img_array, k)
    compressed_uint8 = np.clip(compressed, 0, 255).astype(np.uint8)
    Image.fromarray(compressed_uint8).save(output_path)
    file_size_kb = round(os.stat(output_path).st_size / 1024, 2)
    return file_size_kb

def visualize_results(img_array, s, energy_ratio, k_energy_95, k_elbow, k_psnr_30, 
                     original_size_kb, output_dir='./image'):
    """視覺化分析結果"""
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(18, 12))
    
    # 第一行：4張圖像
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(img_array, cmap='gray')
    ax1.set_title('Original Image', fontsize=11, fontweight='bold')
    ax1.axis('off')
    
    # 95% Energy
    ax2 = plt.subplot(3, 4, 2)
    compressed1 = svd_compress(img_array, k_energy_95)
    psnr1 = calculate_psnr(img_array, compressed1)
    ax2.imshow(compressed1, cmap='gray')
    ax2.set_title(f'95% Energy\nk={k_energy_95}\nPSNR={psnr1:.2f}dB', fontsize=10)
    ax2.axis('off')
    
    # Elbow Method
    ax3 = plt.subplot(3, 4, 3)
    compressed2 = svd_compress(img_array, k_elbow)
    psnr2 = calculate_psnr(img_array, compressed2)
    ax3.imshow(compressed2, cmap='gray')
    ax3.set_title(f'Elbow Method\nk={k_elbow}\nPSNR={psnr2:.2f}dB', fontsize=10)
    ax3.axis('off')
    
    # PSNR>=30dB
    ax4 = plt.subplot(3, 4, 4)
    compressed3 = svd_compress(img_array, k_psnr_30)
    psnr3 = calculate_psnr(img_array, compressed3)
    ax4.imshow(compressed3, cmap='gray')
    ax4.set_title(f'PSNR>=30dB\nk={k_psnr_30}\nPSNR={psnr3:.2f}dB', fontsize=10)
    ax4.axis('off')
    
    # 第二行：奇異值衰減曲線和能量累積曲線
    ax5 = plt.subplot(3, 4, (5, 6))
    ax5.plot(s, 'b-', linewidth=2)
    ax5.axvline(x=k_energy_95, color='g', linestyle='--', linewidth=1.5, 
                label=f'95% Energy (k={k_energy_95})')
    ax5.axvline(x=k_elbow, color='orange', linestyle='--', linewidth=1.5, 
                label=f'Elbow (k={k_elbow})')
    ax5.axvline(x=k_psnr_30, color='r', linestyle='--', linewidth=1.5, 
                label=f'PSNR (k={k_psnr_30})')
    ax5.set_xlabel('Singular Value Index', fontsize=10)
    ax5.set_ylabel('Singular Value Magnitude', fontsize=10)
    ax5.set_title('Singular Value Decay Curve', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    
    ax6 = plt.subplot(3, 4, (7, 8))
    ax6.plot(energy_ratio, 'r-', linewidth=2)
    ax6.axhline(y=0.90, color='b', linestyle=':', linewidth=1, label='90% Threshold')
    ax6.axhline(y=0.95, color='g', linestyle='--', linewidth=1.5, label='95% Threshold')
    ax6.axhline(y=0.99, color='purple', linestyle=':', linewidth=1, label='99% Threshold')
    ax6.axvline(x=k_energy_95, color='g', linestyle='--', linewidth=1.5, alpha=0.5)
    ax6.set_xlabel('K Value', fontsize=10)
    ax6.set_ylabel('Cumulative Energy Ratio', fontsize=10)
    ax6.set_title('Energy Accumulation Curve', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 1.05])
    
    # 第三行：PSNR vs K值曲線和壓縮率比較
    ax7 = plt.subplot(3, 4, (9, 10))
    k_range = range(1, min(200, len(s)), 2)
    psnr_values = [calculate_psnr(img_array, svd_compress(img_array, k)) for k in k_range]
    
    ax7.plot(list(k_range), psnr_values, 'g-', linewidth=2)
    ax7.axhline(y=30, color='orange', linestyle='--', linewidth=1.5, label='30dB Threshold')
    ax7.axhline(y=35, color='r', linestyle='--', linewidth=1.5, label='35dB Threshold')
    ax7.axvline(x=k_psnr_30, color='r', linestyle='--', linewidth=1.5, alpha=0.5)
    ax7.set_xlabel('K Value', fontsize=10)
    ax7.set_ylabel('PSNR (dB)', fontsize=10)
    ax7.set_title('PSNR vs K Value Curve', fontsize=11, fontweight='bold')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)
    
    # 計算真實的壓縮率 (使用實際檔案大小)
    ax8 = plt.subplot(3, 4, (11, 12))
    methods = ['95% Energy', 'Elbow Method', 'PSNR>=30dB']
    k_values = [k_energy_95, k_elbow, k_psnr_30]
    
    # 保存壓縮圖片並計算壓縮率
    compression_ratios = []
    for i, k in enumerate(k_values):
        temp_path = os.path.join(output_dir, f'temp_compressed_{k}.jpg')
        compressed_size = save_compressed_image(img_array, k, temp_path)
        cr_value = cr(original_size_kb, compressed_size)
        compression_ratios.append(cr_value)
    
    colors = ['green', 'orange', 'red']
    bars = ax8.bar(methods, compression_ratios, color=colors, alpha=0.7, edgecolor='black')
    ax8.set_ylabel('Compression Ratio (CR)', fontsize=10)
    ax8.set_title('Compression Ratio Comparison', fontsize=11, fontweight='bold')
    ax8.grid(True, axis='y', alpha=0.3)
    
    # 在柱狀圖上標註數值
    for bar, k_val, ratio in zip(bars, k_values, compression_ratios):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'k={k_val}\nCR={ratio:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('svd_analysis_results.png', dpi=150, bbox_inches='tight')
    print("\n視覺化結果已保存為 'svd_analysis_results.png'")
    plt.show()

def analyze_svd_compression(image_path, output_dir='./image'):
    """完整的SVD壓縮分析"""
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 載入圖像
    img_array = load_image(image_path)
    print(f"圖像尺寸: {img_array.shape}")
    
    # 保存原始灰階圖並獲取大小
    gray_img = np.clip(img_array, 0, 255).astype(np.uint8)
    original_path = os.path.join(output_dir, 'original_gray.jpg')
    Image.fromarray(gray_img).save(original_path)
    original_size_kb = round(os.stat(original_path).st_size / 1024, 2)
    print(f"原始圖像大小: {original_size_kb} KB")
    
    # 執行SVD分解
    U, s, Vt = np.linalg.svd(img_array, full_matrices=False)
    print(f"奇異值數量: {len(s)}")
    
    # 能量保留法
    k_energy_90, energy_ratio = find_optimal_k_energy(s, threshold=0.90)
    k_energy_95, _ = find_optimal_k_energy(s, threshold=0.95)
    k_energy_99, _ = find_optimal_k_energy(s, threshold=0.99)
    
    # 計算真實的壓縮率
    size_90 = save_compressed_image(img_array, k_energy_90, 
                                     os.path.join(output_dir, f'compressed_{k_energy_90}.jpg'))
    size_95 = save_compressed_image(img_array, k_energy_95, 
                                     os.path.join(output_dir, f'compressed_{k_energy_95}.jpg'))
    size_99 = save_compressed_image(img_array, k_energy_99, 
                                     os.path.join(output_dir, f'compressed_{k_energy_99}.jpg'))
    
    cr_90 = cr(original_size_kb, size_90)
    cr_95 = cr(original_size_kb, size_95)
    cr_99 = cr(original_size_kb, size_99)
    
    print(f"\n方法1 - 能量保留法:")
    print(f"  90%能量: k = {k_energy_90}, 檔案大小 = {size_90}KB, CR = {cr_90:.2f}")
    print(f"  95%能量: k = {k_energy_95}, 檔案大小 = {size_95}KB, CR = {cr_95:.2f}")
    print(f"  99%能量: k = {k_energy_99}, 檔案大小 = {size_99}KB, CR = {cr_99:.2f}")
    
    # 肘部法則
    k_elbow = find_optimal_k_elbow(s)
    size_elbow = save_compressed_image(img_array, k_elbow, 
                                        os.path.join(output_dir, f'compressed_{k_elbow}.jpg'))
    cr_elbow = cr(original_size_kb, size_elbow)
    
    print(f"\n方法2 - 肘部法則:")
    print(f"  最佳k = {k_elbow}, 檔案大小 = {size_elbow}KB, CR = {cr_elbow:.2f}")
    
    # PSNR閾值法
    k_psnr_30, psnr_30 = find_optimal_k_psnr(img_array, s, target_psnr=30)
    k_psnr_35, psnr_35 = find_optimal_k_psnr(img_array, s, target_psnr=35)
    
    size_psnr_30 = save_compressed_image(img_array, k_psnr_30, 
                                          os.path.join(output_dir, f'compressed_{k_psnr_30}.jpg'))
    size_psnr_35 = save_compressed_image(img_array, k_psnr_35, 
                                          os.path.join(output_dir, f'compressed_{k_psnr_35}.jpg'))
    
    cr_psnr_30 = cr(original_size_kb, size_psnr_30)
    cr_psnr_35 = cr(original_size_kb, size_psnr_35)
    
    print(f"\n方法3 - PSNR閾值法:")
    print(f"  PSNR>=30dB: k = {k_psnr_30}, 實際PSNR = {psnr_30:.2f}dB, 檔案大小 = {size_psnr_30}KB, CR = {cr_psnr_30:.2f}")
    print(f"  PSNR>=35dB: k = {k_psnr_35}, 實際PSNR = {psnr_35:.2f}dB, 檔案大小 = {size_psnr_35}KB, CR = {cr_psnr_35:.2f}")
    
    # 視覺化結果
    visualize_results(img_array, s, energy_ratio, k_energy_95, k_elbow, k_psnr_30, 
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
        'cr_psnr_35': cr_psnr_35
    }

if __name__ == "__main__":
    image_path = 'matrixaimg.jpg'
    
    results = analyze_svd_compression(image_path)
    print("\n" + "="*60)
    print("建議的K值總結:")
    print("="*60)
    print(f"高壓縮率 (90%能量): k = {results['k_energy_90']}, CR = {results['cr_90']:.2f}")
    print(f"平衡品質 (95%能量): k = {results['k_energy_95']}, CR = {results['cr_95']:.2f} (推薦)")
    print(f"高品質 (99%能量): k = {results['k_energy_99']}, CR = {results['cr_99']:.2f}")
    print(f"肘部法則建議: k = {results['k_elbow']}, CR = {results['cr_elbow']:.2f}")
    print(f"PSNR>=30dB: k = {results['k_psnr_30']}, CR = {results['cr_psnr_30']:.2f}")
    print(f"PSNR>=35dB: k = {results['k_psnr_35']}, CR = {results['cr_psnr_35']:.2f}")
    print("="*60)