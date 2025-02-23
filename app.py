
#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import streamlit as st
from PIL import Image
import os

# Load Model Paths (Make sure these paths are correct!)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
prototxt = os.path.join(BASE_DIR, "models/models_colorization_deploy_v2.prototxt")
model = os.path.join(BASE_DIR, "models/colorization_release_v2.caffemodel")
points = os.path.join(BASE_DIR, "models/pts_in_hull.npy")

# Check if files exist
if not all(map(os.path.exists, [prototxt, model, points])):
    st.error("Model files are missing! Check the 'models' folder.")
    st.stop()

# Load the colorization model
net = cv2.dnn.readNetFromCaffe(prototxt, model)
pts = np.load(points)

# Add the cluster centers as 1x1 convolutions to the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def colorizer(img):
    """ Function to colorize black & white images """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Convert image to LAB color space
    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)

    # Resize to 224x224 for the model
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50  # Mean centering

    # Predict 'ab' channels
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

    # Merge L with predicted 'ab' channels
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    
    # Convert LAB to RGB
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    return colorized

##########################################################################################################

st.write("# Colorizing Black & White Images")
st.write("Upload a black & white image to colorize it!")

file = st.sidebar.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "png"])

if file:
    image = Image.open(file)
    img = np.array(image)

    if img.shape[-1] == 4:  # Convert RGBA to RGB if needed
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    col1, col2 = st.columns(2)
    
    with col1:
        st.text("Original Image")
        st.image(image, use_column_width=True)
    
    with col2:
        st.text("Colorized Image")
        colorized = colorizer(img)
        st.image(colorized, use_column_width=True)

    st.success("✅ Done!")
else:
    st.warning("⚠️ Please upload an image to continue.")
