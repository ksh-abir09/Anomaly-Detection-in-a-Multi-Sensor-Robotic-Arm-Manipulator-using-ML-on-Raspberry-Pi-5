# Anomaly Detection in a Multi-Sensor Robotic Arm/Manipulator using ML on Raspberry Pi 5

## 📌 Overview
This repository contains the implementation of a **real-time anomaly detection and fault diagnosis system** for a robotic arm/manipulator.  
Using multi-sensor data (current, temperature, vibration) from four joints, the system applies **machine learning (ML)** to distinguish **real faults** (mechanical/electrical) from **false alarms** (noise, current spikes, vibration artifacts).

### 🔑 Key Highlights
- **Hardware**: Robotic arm with 4 servo motors, sensors, Raspberry Pi 5, PCA9685 PWM driver.  
- **Data**: 5000+ samples under normal and faulty conditions.  
- **ML**: Comparative evaluation of multiple algorithms (SGD, SVM, Random Forest, Gradient Boosting, MLP).  
- **Deployment**: Models trained offline, exported, and deployed for low-latency inference on Raspberry Pi 5.  

---

## ⚙️ Hardware Setup
- **Robotic Arm/Manipulator** – 4 DOF with servo motors.  
- **Sensors** – 4× current, 4× temperature, 4× vibration.  
- **Controller** – Raspberry Pi 5.  
- **Driver** – PCA9685 PWM driver for servo actuation.  

---

## 🗃️ Data Collection
- **Samples collected**: ~5000 labeled instances.  
- **Normal vs Faulty conditions**: mechanical resistance, overheating, abnormal current draw, and vibration anomalies.  
- **Features extracted**: time-domain (RMS, variance, peaks) + frequency-domain (spectral power, dominant frequency).  

---

## 🤖 Model Training
- **Algorithms compared**:  
  - SGDClassifier (Logistic Regression baseline)  
  - Linear SVM  
  - RBF SVM  
  - Random Forest  
  - HistGradientBoosting  
  - MLPClassifier (small neural network)  

- **Evaluation metrics**: Accuracy, F1-score, AUROC, Precision-Recall, Confusion Matrix.  
- **Cross-validation**: Stratified 5-fold.  
- **Export**: Best model serialized (`.pkl` / `.tflite`) for deployment.  

---

## 🚀 Deployment on Raspberry Pi 5
- Pre-trained models exported from PC.  
- Raspberry Pi 5 runs **real-time inference** while controlling the robotic arm.  
- Optimizations: feature scaling, quantization, batch=1 inference.  
- Results: latency measured in **milliseconds**, suitable for live anomaly detection.  

---

## 📊 Results & Figures
Figures generated from the dataset and training pipeline:  
- Sensor distributions (Normal vs Fault)  
- Feature correlation heatmap  
- Model comparison (Accuracy & F1)  
- Inference latency comparison  
- Confusion matrix (best model)  
- ROC & Precision-Recall curves  

📂 Figures are saved in the [`figures/`](figures/) directory (PDF, SVG, PNG formats).  

---

## 📦 Installation
```bash
# Clone repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
