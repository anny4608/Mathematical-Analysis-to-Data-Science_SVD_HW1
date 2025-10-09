import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'C:\Users\anny4\OneDrive\桌面\SVD_HW1TEST\venv\Lib\site-packages\PyQt5\Qt5\plugins\platforms'

## import 套件
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import math
import pandas as pd
import dataframe_image as dfi

# Create image directory if it doesn't exist
os.makedirs("./image", exist_ok=True)

## 讀取影像跟轉換影像
img_dog = Image.open("matrixaimg.jpg")
img_to_gray = img_dog.convert("L")  # 轉換成灰階
dog_gray = np.array(img_to_gray)
Image.fromarray(dog_gray).save("./image/dog_gray.jpg")

##圖片用矩陣表示
dog = np.array(img_dog)

# 繪圖
plt.figure(figsize=(16 ,10))
plt.subplot(2,3,1)
plt.title(dog.shape)
plt.imshow(dog)
plt.subplot(2,3,2)
plt.title(dog_gray.shape)
plt.imshow(dog_gray, cmap="gray")
plt.subplot(2,3,3)
plt.title(dog.shape) # Corrected from dog_ver
plt.imshow(dog) # Corrected from dog_ver
plt.subplot(2,3,4)
plt.title("R channel")
plt.imshow(dog[:,:,0], cmap="Reds")
plt.subplot(2,3,5)
plt.title("G channel")
plt.imshow(dog[:,:,1], cmap="Greens")
plt.subplot(2,3,6)
plt.title("B channel")
plt.imshow(dog[:,:,2], cmap="Blues")
plt.savefig('svd_hw1_channels.png')
plt.show()

## Metric Functions ##
def norm2 (a, ak):
    return np.linalg.norm(a.astype(np.float64) - ak.astype(np.float64), 2)

def mse (a, ak):
    return np.mean((a.astype(np.float64) - ak.astype(np.float64)) ** 2)

def psnr(originalImg, sampleImg):
    mse_val = mse(originalImg, sampleImg)
    if mse_val < 1.0e-10:
        return 100.0
    return 10 * math.log10(255.0**2 / mse_val)

def cr(originalSize, sampleSize):
    if sampleSize == 0:
        return float('inf')
    return originalSize / sampleSize

def ss(originalSize, sampleSize):
    if originalSize == 0:
        return 0.0
    return (1 - sampleSize / originalSize) * 100


## SVD 分解 ##
def svd_restore(image, k):

    # 對圖形矩陣做 SVD 分解
    u, sigma, v = np.linalg.svd(image, full_matrices=False)
    
    # 避免 K 值超出 sigma 長度
    k = min(len(sigma), k)
    
    # 依照 k 值，得到新的圖形矩陣
    Ak = np.dot(u[:, :k], np.dot(np.diag(sigma[:k]), v[:k, :]))
    
    # 計算誤差指標
    norm_val = norm2(image, Ak)
    mse_val = mse(image, Ak)
    
    # (k+1)-th singular value
    sigma_kp1 = sigma[k] if k < len(sigma) else 0.0
    
    # Clip values to be in the valid range for images [0, 255] and convert to uint8
    Ak = np.clip(Ak, 0, 255)
    Ak = np.rint(Ak).astype(np.uint8)
    
    return Ak, sigma, norm_val, sigma_kp1, mse_val

## Main Execution Block ##

# Image size: 1339, 1145, so 0 < K < 1145 (1 ~ 1144)
rangeArrK = [1, 5, 10, 20, 50, 80, 100, 150, 200, 300, 500, 700, 900, 1000, 1144]

feature_names = ["K值", "2-norm", "sigma k+1", "原圖大小(Kb)", "圖片大小(Kb)", "資料壓縮比(CR)", "節省空間比率(SS)", "均方誤差(MSE)", "峰值訊噪比(PSNR)"]

original_image = dog_gray.astype(np.float64)
fileName = "dog_gray"
original_image_path = f"./image/{fileName}.jpg"
originalImageSize = round(os.stat(original_image_path).st_size / 1024, 2)

results = []

plt.figure(figsize=(16, 24))
p = 0
for k in rangeArrK:
    # SVD decomposition
    G, sigma, norm_val, sigmak1, mse_val = svd_restore(original_image, k)
    
    # Save compressed image
    compressed_image_path = f"./image/{fileName}_{k}.jpg"
    Image.fromarray(G).save(compressed_image_path)
    compressedImageSize = round(os.stat(compressed_image_path).st_size / 1024, 2)
    
    # Calculate metrics
    psnr_val = psnr(original_image, G)
    cr_val = cr(originalImageSize, compressedImageSize)
    ss_val = ss(originalImageSize, compressedImageSize)
    
    results.append([k, norm_val, sigmak1, originalImageSize, compressedImageSize, cr_val, ss_val, mse_val, psnr_val])
    
    # Plot the compressed image
    p += 1
    plt.subplot(5, 3, p)
    title = f"k : {k}"
    plt.title(title)
    plt.imshow(G, cmap="gray")

plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)
plt.savefig('svd_hw1_compressed_images.png')
plt.show()

garyResult = np.array(results)

## Plotting Results ##
def showLinear(table, resize):
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,8),dpi=50,linewidth = 2) # Create figure and two subplots

   # First subplot
   ax1.plot(table[:,0], table[:,6],'s-',color = 'red', label="SS")
   ax1.plot(table[:,0], table[:,7]/resize,'s-',color = 'skyblue', label=f"MSE/{resize}")
   ax1.legend(loc = "best", fontsize=20)
   ax1.set_xlim(0, 80) # Set x-limit for left plot
   ax1.set_ylim(0, 80) # Set y-limit for left plot
   ax1.set_xlabel("k value", fontsize=30, labelpad = 15)
   ax1.set_ylabel("SS / MSE", fontsize=30, labelpad = 15)
   ax1.tick_params(axis='x', labelsize=20)
   ax1.tick_params(axis='y', labelsize=20)


   # Second subplot
   ax2.plot(table[:,0], table[:,4],'s-',color = 'green', label="SIZE")
   ax2.plot(table[:,0], table[:,8],'s-',color = 'blue', label="PSNR")
   ax2.legend(loc = "best", fontsize=20)
   ax2.set_xlim(0, 1200) # Set x-limit for right plot
   ax2.set_ylim(0, 200) # Set y-limit for right plot
   ax2.set_xlabel("k value", fontsize=30, labelpad = 15)
   ax2.set_ylabel("SIZE / PSNR", fontsize=30, labelpad = 15)
   ax2.tick_params(axis='x', labelsize=20)
   ax2.tick_params(axis='y', labelsize=20)

   plt.tight_layout() # Adjust layout to prevent overlapping
   plt.savefig('svd_hw1_linear_plots.png')
   plt.show()

showLinear(garyResult, 50)

## Display Results Table ##
df = pd.DataFrame(garyResult, columns=feature_names)
print(df)
dfi.export(df, 'svd_hw1_results_table.png')

## Plot PSNR and Compression Ratio vs. k ##
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Subplot 1: PSNR vs. k
ax1.plot(garyResult[:, 0], garyResult[:, 8], 'o-')
ax1.set_title('PSNR vs. k')
ax1.set_xlabel('k value')
ax1.set_ylabel('PSNR (dB)')
ax1.grid(True)

# Subplot 2: Compression Ratio vs. k
ax2.plot(garyResult[:, 0], garyResult[:, 5], 'o-')
ax2.set_title('Compression Ratio vs. k')
ax2.set_xlabel('k value')
ax2.set_ylabel('Compression Ratio')
ax2.grid(True)

plt.tight_layout()
plt.savefig('k_vs_psnr_cr.png')
plt.show()
