# Facial Generative methods Evaluation

This repository provides methods to **evaluate GAN inversion and edition** across multiple dimensions, including **quality**, **identity**, and **facial attributes**.  

---



#### TODO  :
```
At the moment, the FAC metric does not distinguish between inversion and editing.
A future release will introduce an option to separate the target attribute change from unintended entanglement effects, used for evaluating facial edits.
 
```

---

## 📑 Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Metrics](#metrics)
- [Options](#options)
- [Directory Structure](#directory-structure)
- [Getting Started](#getting-started)
- [Results](#results)

---

## 🔧 Prerequisites

### Environment
- Python **3.9.1**  
- PyTorch **1.11.0**  
- See `requirements.txt` for the full list of dependencies.  

### Pre-trained Models
Place required checkpoints in `./pretrained_models/`:  
- **SwinFace** (for FIC-Swin): [GitHub link](https://github.com/lxq1000/SwinFace)  
  - `checkpoint_step_79999_gpu_0.pt`  
- **FIQA** (for LFIQ): [GitHub link](https://github.com/LSIbabnikz/AI-KD)  
  - `aikd_diffiqar_model.pth`  
  - `r100.pth`  
- **Landmark detector** (optional, for face alignment): [dlib model](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)  

---

## 📊 Metrics

This framework supports several metrics:  

- **MSE**, **SSIM**, **LPIPS** → standard image quality assessment  
- **LFIQ** → evaluates changes in facial quality scores  
- **FIC** → measures identity preservation using face recognition models  
- **FAC** → evaluates preservation of facial attributes  

---

## ⚙️ Options

The **config file** allows customization of:  
- **Experiment directory**: `main_dir`  
- **Results directory**: where evaluation outputs are stored  
- **Datasets**: works with aligned datasets (CelebA-HQ, ColorFeret, FRL, etc.)  
- **Visualization**: save side-by-side images with metrics overlaid  
- **Statistics**: compute standard deviation in addition to mean
- **Identity-aware datasets**: specify image-to-identity mappings (examples in `./identity_data_info/info/`) 
- If identity information is not available, FIC can still be computed by assigning one image per identity. This works fine for datasets without multiple images per identity.
- **Metric selection**: enable/disable individual metrics  
- **Debug mode**: run evaluation on the first *n* images only  
- **experiment_folder_name**: experiment folder name containing inversion or attributes (see below)

Example below :
```
dir:
  exp_dir: '/data_evaluation_inversion/data_new_code/'
  output_dir: '/data_evaluation_inversion/data_new_code/results/'

# general options
options:
  datasets: ["CelebA_Hq", "Feret", "FRL"]
  image_size: 1024
  save_merged_og_output: true 
  std: true
  multiple_id_datasets: './identity_data_info/info/' 
  experiment_folder_name: 'inversion'

# metrics option
metrics:
  mse: true
  ssim: true
  lpips: true
  lfiq: true
  fic_arcface: true
  fic_swinface: true
  fac: true

#debug options
debug:
  quick_test: 5 
```
---

## 📂 Directory Structure

Your dataset and experiment results should be organized as follows:

```
main_dir/
├─ dataset_A/
│  ├─ aligned/               # Aligned images
│  └─ experiments/
│     ├─ setting_A/
│     │  └─ outputs_1/        # Images for this setting
│     │  └─ .../
│     └─ setting_B/
│        └─ outputs_1/        # Images for this setting
│        └─ .../
├─ dataset_B/
│  ├─ aligned/               # Aligned images
│  └─ experiments/
│     ├─ setting_A/
│     │  └─ outputs_1/        # Images for this setting
│     │  └─ .../
│     └─ setting_B/
│        └─ outputs_1/        # Images for this setting
│        └─ .../

```

**Explanation:**  
- `aligned/` → contains aligned original images  
- `experiments/` → stores different experiment runs  
- `setting_X/outputs_n/` → stores the inverted or edited images for that experiment setting.  
   - Multiple `outputs_n/` folders can exist (e.g., `outputs_smile/`, `outputs_age/`, or `outputs_1/`, `outputs_2/`) if evaluating several attribute edits within the same setting.  
---

## 🚀 Getting Started

### 1. Prepare Data and Pretrained Models
- Download required pretrained weights (see [Prerequisites](#prerequisites))  
- Align images if necessary (see `/align` for alignment utilities)  

### 2. Run Evaluation
``
python eval.py --config_file config/config.yaml
``

---

## 📈 Results  

The evaluation produces **three levels of outputs**:  

1. **Average results per experiment**  
2. **Visualizations of original vs output images**  
3. **Detailed per-image and per-attribute metrics**
4. **FAC Results**

### 1. Results organization  

After running evaluation, the following files and folders are generated in your results directory:  

| File / Folder | Description |
|---------------|-------------|
| `evaluation_results.csv` | Main results table: average metrics per experiment (+STD if enabled) |
| `evaluation_results_attributes.csv` | Attribute-specific results (FAC) |
| `full_metrics/dataset_X/full/*.csv` | Per-image metric values for dataset `X` |
| `full_metrics/dataset_X/attributes/*.csv` | Per-image FAC attribute changes |
| `results/combined/` | Side-by-side visualizations of original vs output images with metrics overlay |


### 2. Combined Images with Metrics  

For each dataset and experiment, side-by-side comparisons are saved under `results/combined/`.  
The metrics for each image are overlaid in the top-left corner for quick inspection.  

![Merged Images](./images/3115.png?raw=true "Merged Images Example")  


### 3. Average Metrics Table  

The file `evaluation_results.csv` summarizes performance across all datasets and experiments.  
Each row corresponds to one experiment setting, with averages (and optionally standard deviations) of all selected metrics.  

![Main Table](./images/main_table.png?raw=true "Main Metrics Table Example")  


### 4. FAC Results  

If **FAC** is enabled, attribute preservation details are available in:  

- `evaluation_results_attributes.csv` → overall per-variant attribute preservation  
- `full_metrics/dataset_X/attributes/*.csv` → per-image FAC attribute changes  

Example of FAC results table:  

![FAC Details](./images/fac_details.png?raw=true "FAC Attribute Table Example")  

---

