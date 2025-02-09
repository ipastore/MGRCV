# Computational Imaging - Camera Pipeline

## 📌 Authors:
- **David Padilla Orenga**
- **Ignacio Pastore Benaim**

---

##  **MATLAB Code Explanation**

This MATLAB pipeline processes a **RAW image (converted to TIFF)** through various stages:  
**Linearization, Demosaicing, White Balancing, Denoising, Color Balancing, Tone Reproduction, and Compression**.  

### 🔹 **Global Variables**
Two global variables control the execution flow:
- **`deploy_all`** → Used for debugging.  
  - Allows saving/loading intermediate `.mat` files to **skip re-running** certain sections.
  - Helps in **testing individual steps** without running the entire pipeline.

- **`plot_all`** → Controls the visualization of results.
  - If **enabled (`1`)**, the script will display **all plots** for intermediate steps.
  - If **disabled (`0`)**, only two figures will appear:  
    - The **final output image**  
    - The **manual white balance selection figure**  
  - When disabled, the script **only applies the selected parameters**, skipping extra test slices.

---

## 🔹 **How to Configure the Pipeline**
Before running the script, **modify the following parameters** in the `CONFIG` section of the MATLAB script:

### ⚙️ **1. Choose the Bayer Pattern**
The demosaicing step requires selecting the correct Bayer pattern of the camera sensor. The available options are:
```matlab
patterns = {'rggb', 'gbrg', 'grbg', 'bggr'};
patternIndex = 1; %  Modify this index to choose the method
```

### ⚙️ **2. Choose the White Balancing Method**
selected_white_balance_method = white_balance_methods{selected_white_balance_methodIndex};

```matlab
white_balance_methods = {'manual', 'gray_world', 'white_world'};
selected_white_balance_methodIndex = 1; % Modify this index to choose the method
```

### ⚙️ **3. Choose Possible Values to Try**

To explore different processing results, you can define multiple test cases:
```matlab
saturation_factors = [1, ...]; % Saturation enhancement levels
porcentage_brighten = [0.25, 0.50, ...];  % Brightness adjustments
gamma_values = [1.7, 1.8, ...];  % Gamma correction options
qualities = [95, 90, 85, ...,];  % JPEG compression qualities
```
If `plot_all = 1`, all test values are applied. 

Note: JPGEs from qualities are always saved.

### ⚙️ **4.  Choose Selected Final Values**

If `plot_all = 0`, the following selected values are used:
```matlab
selected_saturation_factor = 1.50;
selected_porcentage_brighten = 0.75;
selected_gamma = 1.8;
```
Modify these values to change the final image output.

### ⚙️ **5.  Set the input folder**

Before running the pipeline, you must convert RAW files to TIFF format using dcraw.
Run the following command in the default input folder:

```shell
dcraw -4 -D -T 'filename.CR2'
```

The default folder for RAW images is:
```matlab
folderPath = '../data/images_raw';
```

### ⚙️ **6.  Choose the Filename**
Once the RAW file is converted, specify the TIFF filename to 
```matlab
process:filename = 'IMG_0596.tiff'; % Modify this to match your image
```

⚙️ 7. Set the Output Folder

### ⚙️ **6.  Set the output folder**
Processed images will be saved in:

```matlab
output_folder = '../output';
```

Each image will be automatically saved in:

```matlab
../output/filename/operation_type.png;
```

Where:
	•	filename → Matches the input image name.
	•	operation_type → Represents each processing step (e.g., linearized, demosaiced_bilinear, etc.).

This ensures every intermediate and final result is stored efficiently.
