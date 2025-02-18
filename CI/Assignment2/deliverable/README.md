# Computational Imaging - Assignment 2

## 📌 Authors:
- **David Padilla Orenga**
- **Ignacio Pastore Benaim**

---

##  **MATLAB Code Explanation**

This MATLAB script performs various image processing tasks including **blurring, deblurring, and power spectrum analysis** using different apertures and convolution methods.

### 🔹 **Global Variables**
The script uses several global variables to control the execution flow:

- **`sigma_values`** → Array of sigma values for Gaussian noise.
- **`blur_sizes`** → Array of blur sizes for the disk filter.
- **`apertures`** → Cell array of aperture names.
- **`image`** → The input image to be processed.

---

## 🔹 **How to Configure the Script**
Before running the script, **modify the following parameters** in the `CONFIG` section of the MATLAB script:

### ⚙️ **1. Choose Sigma Values**
Define the sigma values for Gaussian noise:
```matlab
sigma_values = [0.001, 0.005, 0.01, 0.02];
```

### ⚙️ **2. Choose Blur Sizes**
Define the blur sizes for the disk filter:
```matlab
blur_sizes = [3, 7, 15];
```

### ⚙️ **3. Define Apertures**
Specify the apertures to be used:
```matlab
apertures = {'zhou', 'raskar', 'Levin', 'circular'};
```

### ⚙️ **4. Choose the Input Image**
Specify the input image to be processed:
```matlab
image = imread('images/penguins.jpg');
```

### ⚙️ **5. Set the Output Folder**
Define the output folder for saving results:
```matlab
output_folder = '../output';
```

---

## 🔹 **Running the Script**
To run the script, simply execute it in MATLAB. The script will iterate over the defined sigma values, blur sizes, and apertures, performing blurring, deblurring, and power spectrum analysis for each combination. The results will be saved in the specified output folder.

---

## 🔹 **Output Structure**
The results will be saved in the following structure:

```plaintext
../output/aperture_name/convolutionType/sigmaX.XXX_blurSizeY/
```
Where:
- `aperture_name` → Name of the aperture used.
- `convolutionType` → Type of convolution method used (e.g., Wiener, Lucy).
- `sigmaX.XXX` → Sigma value used.
- `blurSizeY` → Blur size used.

Each output directory contains processed images and their corresponding power spectrum analysis results. For RGB input images, the script generates three distinct files per process iteration. The naming convention adheres to the following format:

Z_spectrum_image_sigmaX.XXX_blurSizeY.png

Here, Z denotes the color channel, where:
- Z = 1 corresponds to the Red channel,
- Z = 2 corresponds to the Green channel,
- Z = 3 corresponds to the Blue channel.

