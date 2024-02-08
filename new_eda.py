import streamlit as st
import pytesseract
from PIL import Image
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import numpy as np

st.title("OCR App with Exploratory Data Analysis")

# EDA section
eda_options = st.sidebar.selectbox("Data Analysis Options - ", [
    "Word Frequency",
    "Word Length Distribution",
    "Part-of-Speech Tagging",
    "Named Entity Recognition",
    "Custom Analysis (provide Python code)"
])

# OCR section
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    text = pytesseract.image_to_string(image)
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

    elif eda_options == "Part-of-Speech Tagging":
        from nltk import word_tokenize, pos_tag

        # nltk.download('punkt')
        # nltk.download('averaged_perceptron_tagger')
        # nltk.download('universal_tagset')

        tagged_words = pos_tag(word_tokenize(text), tagset='universal')
        pos = np.empty(len(tagged_words), dtype=object)

        for x in range(len(tagged_words)):
            pos[x] = tagged_words[x][1]

        st.subheader("Part-of-Speech Tags")
        unique_values, counts = np.unique(pos, return_counts=True)
        df = pd.DataFrame({"POS Tags": unique_values, "Count": counts})
        st.dataframe(df, hide_index=True)
        st.write(tagged_words)

    elif eda_options == "Named Entity Recognition":

        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        named_entities = [(ent.text, ent.label_) for ent in doc.ents]
        from spacy import displacy
        image_html = displacy.render(doc, style="ent")
        st.markdown(image_html, unsafe_allow_html=True)
        st.subheader("Named Entities")
        st.write(named_entities)

    elif eda_options == "Custom Analysis (provide Python code)":
        # Custom analysis input
        custom_code = st.text_area("Enter Python code for custom analysis (optional)", key="custom_code")
        if custom_code:
            try:
                # Execute custom code with appropriate context
                exec(custom_code, globals(), locals())
            except Exception as e:
                st.error(f"Error in custom code: {e}")
        else:
            st.warning("Please enter Python code for custom analysis.")

    download_text = st.button("Download Text")
    if download_text:
        with open("recognized_text.txt", "w") as f:
            f.write(text)
        st.success("Text downloaded successfully!")
