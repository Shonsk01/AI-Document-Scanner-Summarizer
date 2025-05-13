import streamlit as st
from PIL import Image
import easyocr
import fitz  # PyMuPDF
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk
import io
import time

# Check and download NLTK punkt safely
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set page config
st.set_page_config(page_title="Textify AI - Document OCR & Summarizer", page_icon="📄", layout="centered")

st.title("📄 Textify AI - Document Extractor & Summarizer")

# OCR Reader Initialization
reader = easyocr.Reader(['en'])

# Extractive summarization function using Sumy
def extractive_summary(text, num_sentences=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

# OCR extraction from image
def ocr_image(img):
    st.info("Performing OCR on Image...")
    result = reader.readtext(img)
    extracted_text = "\n".join([text for _, text, _ in result])
    return extracted_text

# OCR extraction from PDF
def ocr_pdf(pdf_file):
    st.info("Extracting text from PDF...")
    pdf_doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in pdf_doc:
        text += page.get_text()
    return text

# File upload
uploaded_file = st.file_uploader("Upload a PDF or Image file", type=["pdf", "png", "jpg", "jpeg"])

# Custom CSS for progress bar below button
st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Button and processing
if uploaded_file:
    if st.button("🧾 Extract & Summarize"):
        progress_bar = st.progress(0)
        time.sleep(0.5)
        progress_bar.progress(10)

        # Read file
        if uploaded_file.type == "application/pdf":
            text = ocr_pdf(uploaded_file)
        else:
            image = Image.open(uploaded_file)
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            text = ocr_image(img_bytes)

        progress_bar.progress(50)

        # Summarization
        if text.strip():
            summary_text = extractive_summary(text, num_sentences=10)
            progress_bar.progress(100)
            st.success("✅ Summary Ready!")
            st.subheader("📃 Extracted Text")
            st.write(text)

            st.subheader("🔍 Extractive Summary")
            st.write(summary_text)
        else:
            st.error("❌ Could not extract any text.")
            progress_bar.progress(0)
    else:
        st.info("Click the button to start OCR and Summarization.")
