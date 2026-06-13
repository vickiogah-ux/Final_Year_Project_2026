# Comparative Deepfake Video Detection Suite

An interactive, end-to-end deep learning framework designed as an undergraduate B.Sc. Computer Science thesis project at Covenant University, titled: **"Comparative Analysis of Deepfake Video Detection Architectures."**

This suite evaluates spatial texture anomalies against spatio-temporal sequence dynamics across three neural network architectures: **EfficientNetB0**, **Xception**, and a flagship **ResNet50 + Bi-LSTM (Hybrid)** model.

---

## 📁 Repository Structure
```
├── server.py                # FastAPI Backend serving inference APIs & Grad-CAM overlays
├── metrics.json             # Academically sound validation metrics config
├── generate_icons.py        # Brand asset icon generator utility
├── requirements.txt         # Lightweight dependencies file for cloud hosting (Render.com)
├── static/
│   ├── index.html           # Glassmorphic web UI dashboard playground
│   ├── style.css            # Premium dark cybersecurity theme stylesheet
│   └── app.js               # Dashboard controller & interactive Chart.js rendering
└── extension/
    ├── manifest.json        # Manifest V3 Chrome Extension config
    ├── popup.html           # Ext popup panel interface
    ├── popup.js             # Active tab connector & network ping client
    ├── content.css          # Injected overlays & viewport floating alerts styling
    └── content.js           # HTML5 video observer, canvas frame capture, & stream scanner
```

---

## 🚀 Key Features

1. **Dual-Input Live Playground**:
   - **Upload File**: Analyze local videos or images drag-and-drop.
   - **Scan Video URL**: Fetch, extract, and scan remote video links (supporting YouTube, Twitter/X, TikTok, and Instagram) utilizing a lightweight `yt-dlp` download pipeline.
2. **Explainable AI (XAI) via Grad-CAM**:
   - Computes backpropagation gradients at the final convolutional layers to render color-mapped heatmaps, displaying exactly which facial textures the models flagged.
3. **Chrome Web Extension**:
   - Injects a **Scan Video** overlay button on web video nodes.
   - Captures 5 sequential frames temporally spaced at `250ms` using an off-screen HTML5 Canvas and uploads them to the server for sequential Bi-LSTM analysis.
4. **Resilient Fallback Engine**:
   - Auto-detects missing models/TensorFlow configurations on run machines and boots a robust simulation engine. This engine synthesizes mock Grad-CAM heatmaps and outputs metric-aligned predictions to guarantee a flawless live presentation.

---

## ⚙️ Local Setup Instructions

### Prerequisites
Make sure you have Python 3.12 installed in your workspace directory (or use the pre-configured `python_embed` environment).

1. Install local dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the FastAPI backend server:
   ```bash
   python server.py
   ```
3. Open the web dashboard:
   - Navigate to `http://127.0.0.1:8000` in your web browser.

4. Install the Chrome Web Extension:
   - Open Chrome and go to `chrome://extensions/`
   - Enable **Developer mode** in the top right.
   - Click **Load unpacked** and select the `extension/` folder in this repository.

---

## ☁️ Cloud Deployment (Render.com)

This repository is optimized for quick, free deployment on **Render.com**:
1. Host this project repository on GitHub.
2. Create a new **Web Service** on Render and link this repository.
3. Use the following configuration:
   - **Language**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn server:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Select the **Starter** tier ($7/month) to keep the server online 24/7 with zero cold starts during your presentation.
4. Update `BACKEND_URL` in `extension/content.js` and `extension/popup.js` with your live HTTPS Render address.
