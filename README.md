🌍 UrbanNet
AI-Powered Satellite Building Detection & Urban Footprint Extraction System

UrbanNet is an end-to-end geospatial artificial intelligence pipeline designed for automated building footprint extraction from multispectral satellite imagery.

The system integrates cloud-based remote sensing, supervised machine learning, and deep learning–based semantic segmentation to enable scalable urban infrastructure analysis.

Developed using Sentinel-2 imagery (2023) over Delhi NCR, UrbanNet demonstrates how modern AI techniques can be fused with GIS workflows for high-resolution urban mapping and spatial analytics.

🚀 Project Overview

UrbanNet implements a three-stage AI-driven geospatial workflow:

Cloud-Based Multispectral Feature Engineering (Google Earth Engine)

Supervised Machine Learning Classification (Random Forest)

Deep Learning Semantic Segmentation (U-Net, PyTorch)

GIS-Based Spatial Analysis and Vectorization (QGIS)

The system moves beyond traditional spectral thresholding by integrating:

Spectral indices (NDVI, NDWI, NDBI)

Texture-based spatial features (GLCM)

Ensemble classification

Convolutional Neural Network segmentation

The result is a scalable and reproducible pipeline for automated urban building detection.

🧠 Technical Architecture
🔹 Stage 1 — Geospatial Data Processing (Google Earth Engine)

Sentinel-2 Surface Reflectance (COPERNICUS/S2_SR_HARMONIZED)

Year: 2023

Cloud Filtering: < 10%

Median Composite Generation

Feature Engineering

NDVI (Vegetation separation)

NDWI (Water separation)

NDBI (Built-up enhancement)

GLCM Texture (Spatial variability enhancement)

A multi-dimensional feature stack was constructed to improve class separability in complex urban environments.

🔹 Stage 2 — Machine Learning Classification

Algorithm: Random Forest
Trees: 70
Input Features: 9 spectral & derived features
Classes: Buildings, Vegetation, Water, Roads

Performance Metrics

Overall Accuracy: 93.7%

Kappa Coefficient: 0.91+

Random Forest provided efficient, cloud-scalable land cover classification and served as a baseline for comparison against deep learning.

🔹 Stage 3 — Deep Learning Semantic Segmentation

Framework: PyTorch
Architecture: U-Net
Task: Binary segmentation (Building vs Background)
Patch Size: 256 × 256
Total Samples: 897

Training Configuration

Loss Function: Binary Cross Entropy (BCE)

Optimizer: Adam

Learning Rate: 0.001

Epochs: 8

Batch Size: 8

Evaluation Metrics

Intersection over Union (IoU)

Dice Score

Validation Loss

The U-Net model improved boundary precision and pixel-level building delineation compared to pixel-based classification.

🔹 Stage 4 — GIS Spatial Analysis (QGIS)

Raster → Vector Conversion (Polygonize)

Built-up Area Calculation

Urban Density Estimation

Cartographic Map Layout

This stage translated AI outputs into actionable geospatial analytics.

📊 Key Results
Component	Performance
Random Forest Accuracy	93.7%
Deep Learning IoU	High Overlap
Built-up Area Extraction	Automated vector output
Output Format	GIS-ready shapefiles

UrbanNet demonstrates that hybrid ML + DL approaches significantly improve spatial precision compared to traditional classification methods.

🛠 Tech Stack
Geospatial

Google Earth Engine (JavaScript API)

Sentinel-2 MSI

QGIS

Machine Learning

Random Forest (GEE)

Deep Learning

PyTorch

Custom U-Net Architecture

NumPy

Rasterio

Matplotlib

📂 Repository Structure
UrbanNet/
│
├── README.md
├── requirements.txt
│
├── gee/
│   └── urban_net_workflow.js
│
├── deep_learning/
│   ├── dataset.py
│   ├── model_unet.py
│   ├── train.py
│   ├── metrics.py
│   └── predict.py
│
├── results/
│   ├── rf_classification.png
│   ├── confusion_matrix.png
│   ├── training_curve.png
│   └── segmentation_output.png
│
└── report/
    └── UrbanNet_Project_Report.pdf

🔍 Applications

UrbanNet can be extended for:

Smart city infrastructure mapping

Urban growth monitoring

Disaster damage assessment

Informal settlement detection

Climate-resilient urban planning

Infrastructure density estimation

📈 Future Enhancements

Incorporation of higher-resolution imagery (PlanetScope, WorldView)

Transfer learning using pre-trained segmentation backbones

Multi-temporal urban growth analysis

Instance segmentation (Mask R-CNN)

Deployment as a web-based urban analytics dashboard

📜 License

This project is licensed under the MIT License.

👤 Author

Sarthak Mehta
B.Tech Computer Science & Engineering
Dr. B.R. Ambedkar National Institute of Technology Jalandhar
