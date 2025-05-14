import streamlit as st
from PIL import Image
import easyocr
import fitz  # PyMuPDF
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
import nltk
import io

# ---- FIX FOR NLTK TOKENIZER (STREAMLIT CLOUD COMPATIBLE) ----
import os

# Add /tmp/nltk_data to path always (safe for Streamlit Cloud)
NLTK_DATA_PATH = "/tmp/nltk_data"
nltk.data.path.append(NLTK_DATA_PATH)

# Ensure punkt is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=NLTK_DATA_PATH)

# ---- INIT EASYOCR ----
reader = easyocr.Reader(['en'])

# ---- STREAMLIT STYLING ----
st.set_page_config(page_title="Textify - AI Document Extractor & Summarizer", layout="wide")
st.markdown(
    """
    <style>
        .main {background-color: #f8f9fa;}
        .title {font-size: 36px; color: #1c1c1c; font-weight: bold;}
        .subtitle {font-size: 20px; color: #007bff;}
        .upload-section {padding: 30px; background-color: #ffffff; border-radius: 10px; box-shadow: 0px 4px 10px rgba(0,0,0,0.1);}
        .summary-section {padding: 30px; background-color: #f1f8ff; border-radius: 10px; box-shadow: 0px 4px 10px rgba(0,0,0,0.1);}
        .text-area {border-radius: 10px; background-color: #f8f9fa; font-size: 14px; color: #333333;}
        .button {background-color: #007bff; color: white; font-weight: bold; border-radius: 5px; padding: 10px 20px; margin-top: 20px;}
        .spinner {color: #007bff;}
    </style>
    """, unsafe_allow_html=True)

# ---- HEADER ----
st.markdown('<h1 class="title">📄 Textify - by Shon Sudhir Kamble</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered Document Extractor & Summarizer</p>', unsafe_allow_html=True)

# ---- FILE UPLOAD ----
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Image or PDF", type=["png", "jpg", "jpeg", "pdf"])
st.markdown('</div>', unsafe_allow_html=True)

# ---- TEXT EXTRACTION FUNCTIONS ----
def extractive_summary(text, num_sentences=10):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return "\n".join([f"• {sentence}" for sentence in summary])

def extract_text_from_pdf(pdf_file):
    pdf_reader = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in pdf_reader:
        text += page.get_text()
    return text

# ---- MAIN LOGIC ----
if uploaded_file:
    file_name = uploaded_file.name.lower()
    st.write(f"File uploaded: {file_name}")

    if file_name.endswith(".pdf"):
        st.write("Processing PDF...")
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)
        st.success("✅ Text extracted from PDF.")
    else:
        st.write("Processing Image...")
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption=f'Uploaded Image: {file_name}', use_container_width=True)

            image_bytes = io.BytesIO()
            image.save(image_bytes, format='PNG')
            image_bytes = image_bytes.getvalue()

            with st.spinner("Extracting text from Image..."):
                result = reader.readtext(image_bytes, detail=0, paragraph=True)
                text = "\n".join(result)
            st.success("✅ Text extracted from Image.")
        except Exception as e:
            st.error(f"❌ Error while processing image: {e}")
            st.write(e)

    # ---- DISPLAY EXTRACTED TEXT & SUMMARY ----
    if text.strip():
        st.markdown('<div class="summary-section">', unsafe_allow_html=True)
        st.subheader("📜 Extracted Text")
        st.text_area("", text, height=300, key="extracted_text", disabled=True, label_visibility="collapsed")

        if st.button("📋 Fast Summarize", key="summarize_button", help="Generate summary from extracted text"):
            with st.spinner("Generating summary using extractive method..."):
                summary_text = extractive_summary(text, num_sentences=10)
            st.success("✅ Summary Ready!")
            st.subheader("📝 Summary")
            st.text_area("", summary_text, height=300, key="summary_text", disabled=True, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠ No text found to process.")
