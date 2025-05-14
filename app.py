import streamlit as st
from PIL import Image
import easyocr
import fitz  # PyMuPDF
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
import nltk
import io
import numpy as np

# Ensure NLTK resources are present
nltk.download('punkt', quiet=True)

# Initialize OCR reader (only English for lightweight model)
reader = easyocr.Reader(['en'], gpu=False)

# Title
st.title("📄 Textify - AI Doc Extractor & Summarizer")
st.caption("by Shon Sudhir Kamble")

# File uploader
uploaded_file = st.file_uploader("Upload an Image (PNG/JPG) or PDF", type=["png", "jpg", "jpeg", "pdf"])

# Summarization function
def extractive_summary(text, num_sentences=8):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return "\n".join([f"• {sentence}" for sentence in summary])

# PDF extractor
def extract_text_from_pdf(pdf_file):
    pdf_reader = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in pdf_reader:
        text += page.get_text()
    return text

# Image OCR
def extract_text_from_image(uploaded_image):
    try:
        image = Image.open(uploaded_image).convert("RGB")
        # Convert PIL image to NumPy array (required by EasyOCR)
        img_array = np.array(image)
        result = reader.readtext(img_array, detail=0, paragraph=True)
        return "\n".join(result)
    except Exception as e:
        st.error(f"Failed to process the image: {e}")
        return ""

# Processing logic
if uploaded_file:
    file_name = uploaded_file.name.lower()
    st.info(f"📂 File uploaded: {file_name}")

    if file_name.endswith(".pdf"):
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)
        st.success("✅ Text extracted from PDF.")
    else:
        with st.spinner("Extracting text from Image..."):
            text = extract_text_from_image(uploaded_file)
        st.success("✅ Text extracted from Image.")

    # Display extracted text
    if text.strip():
        st.subheader("📜 Extracted Text")
        st.text_area("Extracted Text", text, height=300)

        if st.button("📋 Fast Summarize"):
            with st.spinner("Generating summary..."):
                summary_text = extractive_summary(text)
            st.success("✅ Summary R
