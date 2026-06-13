import tensorflow as tf

print("Loading Hybrid Model...")
try:
    # compile=False saves time and prevents metric-related loading errors
    hybrid = tf.keras.models.load_model('hybrid_best_model.h5', compile=False)
    
    print("\n✅ SUCCESS! Model loaded.")
    print("=========================================")
    print(f"EXPECTED INPUT SHAPE: {hybrid.input_shape}")
    print("=========================================\n")
    
except Exception as e:
    print("\n❌ Error loading model:", e)