# SVD Image Compression Analysis

## NCHU SVD HW1
This project is a homework assignment for a Singular Value Decomposition (SVD) course at NCHU.

This project demonstrates the use of Singular Value Decomposition (SVD) for image compression. It includes Python scripts to perform the compression, analyze the results, and visualize the effects of different compression levels.

## File Descriptions

- `svd_hw1.py`: The main script for performing SVD image compression. It takes an image, applies SVD with different `k` values, and saves the compressed images.
K_select.py A script to analyze the relationship between the `k` value in SVD, the Peak Signal-to-Noise Ratio (PSNR), and the Compression Ratio (CR).
- `image/`: This directory contains the original and compressed images.
- `svd_analysis_results.png`, `svd_hw1_channels.png`, `svd_hw1_compressed_images.png`, `svd_hw1_linear_plots.png`, `svd_hw1_results_table.png`, `k_vs_psnr_cr.png`: These are output files containing plots and tables that visualize the results of the SVD analysis.

## How to Run

To run this project, you need to have Python and pip installed.

Create and activate a virtual environment:

```bash
# Create the virtual environment
python -m venv venv

# Activate the virtual environment (on Windows)
.\venv\Scripts\activate
```

Install the required packages:

```bash
# Install all the necessary packages from the environment.txt file
pip install -r environment.txt
```

Run the script:

```bash
# Execute the main script
python svd_hw1.py
```

## Results

The scripts generate several output files:

-   **Compressed Images:** The `image/` directory will contain the compressed versions of the original image, named according to the `k` value used.
-   **Analysis Plots:** The `.png` files provide visualizations of the analysis, including:
    *   The relationship between `k`, PSNR, and CR.
    *   The distribution of singular values.
    *   A comparison of the original and compressed images.
