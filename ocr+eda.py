import streamlit as st
import pytesseract
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

st.title("OCR App with Exploratory Data Analysis")

# EDA section
eda_options = st.sidebar.selectbox("Exploratory Data Analysis", ["Word Frequency", "Word Length Distribution"])

# OCR section
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    text = pytesseract.image_to_string(image)  # Directly use PyTesseract

    st.write("Recognized text:")
    st.write(text)

    data = pd.Series(text.split())

    if eda_options == "Word Frequency":
        st.subheader("Word Frequency Distribution")
        freq_dist = data.value_counts()
        st.bar_chart(freq_dist)
    elif eda_options == "Word Length Distribution":
        st.subheader("Word Length Distribution")
        word_lengths = data.str.len()
        st.bar_chart(word_lengths.value_counts())

    download_text = st.button("Download Text")
    if download_text:
        with open("recognized_text.txt", "w") as f:
            f.write(text)
        st.success("Text downloaded successfully!")
