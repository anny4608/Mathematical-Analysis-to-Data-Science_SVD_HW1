<<<<<<< HEAD
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
=======
# NCHU SVD HW1

This project is a homework assignment for a Singular Value Decomposition (SVD) course at NCHU.

## Objective

The main objectives of this assignment are:

1.  To use the image `matrixaimg.jpg` for SVD analysis.
2.  To empirically prove the property that the 2-norm of the error matrix (`||A - Ak||_2`) is equal to the `(k+1)`-th singular value (`sigma_{k+1}`).
3.  To decompose the image using SVD and reconstruct it with different numbers of singular values (`k`).
4.  To calculate and analyze the following metrics for different `k` values:
    *   Mean Squared Error (MSE)
    *   Compression Ratio (CR)
    *   Space Saved (SS)
    *   2-norm of the error
    *   `(k+1)`-th singular value (`sigma_{k+1}`)
    *   Image size (original and compressed)
    *   Peak Signal-to-Noise Ratio (PSNR)
5.  To present all the calculated metrics in a clear and organized table.

## How to Run the Code

To run this project, you need to have Python and `pip` installed.

1.  **Create and activate a virtual environment:**
    ```bash
    # Create the virtual environment
    python -m venv venv

    # Activate the virtual environment (on Windows)
    .\venv\Scripts\activate
    ```

2.  **Install the required packages:**
    ```bash
    # Install all the necessary packages from the environment.txt file
    pip install -r environment.txt
    ```

3.  **Run the script:**
    ```bash
    # Execute the main script
    python svd_hw1.py
    ```

## Output

The script will generate the following output files:

*   `svd_hw1_channels.png`: A plot showing the original image and its R, G, B channels.
*   `svd_hw1_compressed_images.png`: A plot showing the reconstructed images for different `k` values.
*   `svd_hw1_linear_plots.png`: Plots showing the relationship between `k` and various metrics (PSNR, MSE, etc.).
*   `k_vs_psnr_cr.png`: A plot showing the relationship between PSNR and Compression Ratio vs. `k`.
*   `svd_hw1_results_table.png`: An image of the final table containing all the calculated metrics.
*   The script will also print the results table to the console.

## Code Description

The main script `svd_hw1.py` performs the following steps:

1.  Loads the `matrixaimg.jpg` image and converts it to grayscale.
2.  Defines several functions to calculate the required metrics (MSE, PSNR, etc.).
3.  Defines a function to perform SVD and reconstruct the image for a given `k`.
4.  Loops through a range of `k` values, and for each `k`:
    *   Reconstructs the image.
    *   Calculates all the metrics.
    *   Saves the reconstructed image.
5.  Generates and saves plots to visualize the results.
6.  Creates a pandas DataFrame to display the results in a table, and saves the table as an image.
>>>>>>> 2fae496f30cf45800a264073140ef51d55b162d5
