import os
os.environ.pop('TF_USE_LEGACY_KERAS', None) # Guarantee legacy mode is completely dead

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tempfile
import tensorflow as tf
import keras

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Deepfake Detection Interface", layout="wide")
st.title("🛡️ Deepfake Detection & Comparative Analysis")
st.markdown("Upload a video or image. The system utilizes Spatial (Xception/EfficientNet) and Spatio-Temporal (ResNet+BiLSTM) models.")

# --- THE ARCHITECTURE REBUILD (The Bug Shield) ---
def build_hybrid_architecture(sequence_length=5):
    base_model = keras.applications.ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
    
    model = keras.models.Sequential()
    
    # 🛑 THE FIX: An explicit Input layer prevents TimeDistributed from parsing corrupted strings
    model.add(keras.layers.Input(shape=(sequence_length, 224, 224, 3)))
    
    # Now TimeDistributed just inherits the clean shape from above natively
    model.add(keras.layers.TimeDistributed(base_model))
    model.add(keras.layers.TimeDistributed(keras.layers.GlobalAveragePooling2D()))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=False)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    
    return model

# --- LOAD AI BRAINS ---
@st.cache_resource
def load_models():
    # Load the Spatial Models normally
    xception = keras.models.load_model('xception_best_model.h5', compile=False)
    efficientnet = keras.models.load_model('efficientnet_best_model.h5', compile=False)
    
    # Build the bug-free architecture in memory
    hybrid = build_hybrid_architecture(sequence_length=5)
    
    # Inject the weights directly from your rescued file!
    hybrid.load_weights('hybrid_model_fixed.keras')
    
    # Crash-proof OpenCV Face Detector
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    return xception, efficientnet, hybrid, detector

with st.spinner("Booting Pure Keras 3 AI Pipeline..."):
    xception_model, efficientnet_model, hybrid_model, face_detector = load_models()

# --- THE BULLETPROOF GRAD-CAM ---
def make_gradcam_heatmap(img_array, model, pred_score):
    target_layer = None
    valid_layer_types = (keras.layers.Conv2D, keras.layers.DepthwiseConv2D, keras.layers.SeparableConv2D)
    
    for layer in reversed(model.layers):
        if isinstance(layer, valid_layer_types):
            target_layer = layer.name
            break
            
    if target_layer is None:
        return np.zeros((224, 224))

    grad_model = keras.models.Model(
        inputs=[model.inputs], 
        outputs=[model.get_layer(target_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        tape.watch(inputs)
        model_outputs = grad_model(inputs)
        
        conv_outputs = model_outputs[0]
        predictions = model_outputs[1]
        
        if isinstance(conv_outputs, list): conv_outputs = conv_outputs[0]
        if isinstance(predictions, list): predictions = predictions[0]
            
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None: return np.zeros((224, 224))
    
    if pred_score < 0.5: grads = -grads

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_heat = tf.math.reduce_max(heatmap)
    
    if max_heat == 0: return np.zeros(heatmap.shape)
    return (heatmap / max_heat).numpy()

def display_heatmap(image, heatmap):
    if np.max(heatmap) == 0:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    jet_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    jet_heatmap_rgb = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(jet_heatmap_rgb, 0.4, image, 0.6, 0)

# --- VIDEO EXTRACTION LOGIC ---
def extract_video_sequence(video_path, num_frames=5):
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = vidcap.read()
        if success:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    vidcap.release()
    return frames

# --- THE UPLOAD UI ---
uploaded_file = st.file_uploader("Upload a face image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file is not None:
    is_video = uploaded_file.name.split('.')[-1].lower() in ['mp4', 'avi', 'mov']
    img_array = None
    video_frames = None
    
    if is_video:
        st.info("🎥 Video detected! Extracting 5-frame temporal sequence...")
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        temp_filepath = tfile.name
        tfile.close() 
        
        video_frames = extract_video_sequence(temp_filepath, num_frames=5)
        
        try: os.remove(temp_filepath)
        except: pass
        
        if len(video_frames) == 5:
            img_array = video_frames[2] 
            st.image(Image.fromarray(img_array), caption='Middle Frame (Spatial Target)', width=300)
        else:
            st.error("Error: Could not extract enough frames.")
            st.stop()
            
    else:
        image_display = Image.open(uploaded_file)
        st.image(image_display, caption='Uploaded Raw Image', width=300)
        img_array = np.array(image_display.convert('RGB'))

    # --- THE ANALYSIS BUTTON ---
    if img_array is not None and st.button("Analyze with AI Pipeline"):
        with st.spinner("Executing Multi-Model Analysis..."):
            
            gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) == 0:
                st.error("🚨 No face detected! Please upload a clearer image/video.")
            else:
                x, y, width, height = faces[0]
                padding = 20
                start_x, start_y = max(0, x - padding), max(0, y - padding)
                end_x, end_y = min(img_array.shape[1], x + width + padding), min(img_array.shape[0], y + height + padding)
                
                cropped_face = img_array[start_y:end_y, start_x:end_x]
                img_resized = cv2.resize(cropped_face, (224, 224))
                img_float32 = img_resized.astype(np.float32) 
                
                # Spatial Preprocessing
                img_xcp = np.expand_dims(img_float32 / 255.0, axis=0)
                img_eff = np.expand_dims(img_float32, axis=0)
                
                # 1. Spatial Predictions
                xcp_pred = xception_model.predict(img_xcp, verbose=0)[0][0]
                eff_pred = efficientnet_model.predict(img_eff, verbose=0)[0][0]

                # 2. Hybrid Spatio-Temporal Prediction
                hybrid_pred = None
                if is_video and video_frames:
                    seq_faces = []
                    for frame in video_frames:
                        f_crop = frame[start_y:end_y, start_x:end_x]
                        f_res = cv2.resize(f_crop, (224, 224)).astype(np.float32) / 255.0
                        seq_faces.append(f_res)
                    
                    hybrid_input = np.expand_dims(np.array(seq_faces), axis=0)
                    hybrid_pred = hybrid_model.predict(hybrid_input, verbose=0)[0][0]

                # 3. Generate Heatmaps
                xcp_heatmap = make_gradcam_heatmap(img_xcp, xception_model, xcp_pred)
                eff_heatmap = make_gradcam_heatmap(img_eff, efficientnet_model, eff_pred)
                xcp_overlay = display_heatmap(cropped_face, xcp_heatmap)
                eff_overlay = display_heatmap(cropped_face, eff_heatmap)
                
                # --- DISPLAY RESULTS ---
                st.markdown("---")
                st.subheader("Comparative Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### XceptionNet")
                    st.caption("Spatial Baseline")
                    if xcp_pred < 0.5: st.error(f"🚨 FAKE ({(1 - xcp_pred):.2%})")
                    else: st.success(f"✅ REAL ({xcp_pred:.2%})")
                    st.image(Image.fromarray(xcp_overlay), caption="Xception Attention", use_column_width=True)
                        
                with col2:
                    st.markdown("### EfficientNetB0")
                    st.caption("Spatial Optimized")
                    if eff_pred < 0.5: st.error(f"🚨 FAKE ({(1 - eff_pred):.2%})")
                    else: st.success(f"✅ REAL ({eff_pred:.2%})")
                    st.image(Image.fromarray(eff_overlay), caption="EfficientNet Attention", use_column_width=True)
                
                with col3:
                    st.markdown("### ResNet50 + Bi-LSTM")
                    st.caption("Spatio-Temporal Analysis")
                    if not is_video:
                        st.warning("⚠️ Hybrid Model Requires Video Input.")
                        st.info("Upload an mp4/avi file to analyze temporal movement and jitter.")
                    else:
                        if hybrid_pred < 0.5: st.error(f"🚨 FAKE ({(1 - hybrid_pred):.2%})")
                        else: st.success(f"✅ REAL ({hybrid_pred:.2%})")
                        st.markdown("> *Analyzes sequential jitter and blending anomalies across 5 frames using Bi-Directional LSTM tracking.*")