# Vision-Based Cargo Load Optimization System

![Screenshot 2025-05-01 152712](https://github.com/user-attachments/assets/b61ba9c0-038e-41b4-81d7-b8cd56130971)

## 📌 Overview
A Python-based system that combines computer vision and 3D bin packing algorithms to optimize cargo loading. Processes product images to extract dimensions and generates optimal container loading plans with 3D visualization.

## ✨ Key Features
- **Image Processing**: Extracts item dimensions from 2D images
- **Smart Packing**: LightningPacker algorithm with 75%+ utilization
- **3D Visualization**: Interactive container visualization
- **High Performance**: Processes 5,000+ images in minutes
- **Dynamic Optimization**: Adjusts container sizes automatically

## 🛠️ Installation
```bash
pip install -r requirements.txt
```
🚀 Usage
Configure image directory in App.py

```Run the system
python App.py
```

View packing results and 3D visualizations

##📂 Dataset Preparation
- Uses the Amazon Bin Image Dataset:

- Place images in /input_images folder

- Supported formats: JPG, JPEG, PNG

##🧠 Core Algorithms
Component	Technique
- Image Processing	Adaptive Thresholding + Contour Detection

- Depth Estimation	Solidity-based Heuristic

- Packing Algorithm	Hybrid Spatial Grid + Fast-Track Placement

- Optimization	Dynamic Container Resizing


##📊 Performance Metrics
- Metric	Value

- Image Processing Speed	5000 images in <10 mins

- Container Utilization	75-85%

- Collision Checks	98.1% reduction
