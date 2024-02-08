import streamlit as st
import pytesseract
from PIL import Image

st.title("OCR App")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    ocr_process = st.button("Convert to text")
    if ocr_process:
        text = pytesseract.image_to_string(image)

        st.write("Recognized text:")
        st.write(text)

        download_text = st.button("Download Text")
        if download_text:
            with open("recognized_text.txt", "w") as f:
                f.write(text)
            st.success("Text downloaded successfully!")
