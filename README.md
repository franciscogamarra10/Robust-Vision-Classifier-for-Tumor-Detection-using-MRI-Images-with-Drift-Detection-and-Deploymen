# Robust-Vision-Classifier-for-Tumor-Detection-using-MRI-Images-with-Drift-Detection-and-Deploymen
This project provides a robust vision pipeline for tumor detection in MRI scans. 
It integrates state‑of‑the‑art deep learning models with drift detection metrics (Wasserstein, KL divergence, PIA) 
to ensure reliability when test data differs from training distribution (e.g., illumination, contrast, acquisition variability).
The repository is structured for end‑to‑end experimentation and deployment:
Experimentation folder for reproducible training and drift analysis.
Deployment stack with Docker + Redis + Render for scalable inference.
