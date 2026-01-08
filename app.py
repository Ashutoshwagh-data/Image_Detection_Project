Subject: (not an email — ignore)

from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np

# Page title

st.title("YOLO Object Detection App")

# Load YOLO model

model = YOLO("yolo11n.pt")

# Upload image

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
# Read image
image = Image.open(uploaded_file)
st.image(image, caption="Uploaded Image", use_column_width=True)

```
# Run YOLO model
results = model(image)

# Show result
result_image = results[0].plot()
st.image(result_image, caption="Detected Objects", use_column_width=True)
```
