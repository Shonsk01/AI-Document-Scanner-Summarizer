import streamlit as st
from PIL import Image
import easyocr
import fitz  # PyMuPDF
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
import nltk
import io

# Ensure NLTK resources are present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Initialize OCR reader
reader = easyocr.Reader(['en'])

# Streamlit UI Styling
st.set_page_config(page_title="Textify - AI Document Extractor & Summarizer", layout="wide")
st.markdown(
    """
    <style>
        body {background-color: #f0f2f6;}
        .title {font-size: 40px; color: #222831; font-weight: 700; margin-bottom: 5px;}
        .subtitle {font-size: 18px; color: #393E46; margin-top: 0px;}
        .section {padding: 20px 30px; background-color: #ffffff; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);}
        .upload-label {font-size: 16px; font-weight: 600; color: #30475E;}
        .stButton button {background-color: #30475E; color: #FFFFFF; font-weight: 600; border-radius: 8px;}
        .stTextArea textarea {border-radius: 10px; background-color: #f8f9fa; font-size: 14px;}
    </style>
    """, unsafe_allow_html=True)

# Title and introduction
st.markdown('<h1 class="title">📄 Textify</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered Document Extractor & Summarizer by Shon Sudhir Kamble</p>', unsafe_allow_html=True)

# File uploader section
st.markdown('<div class="section">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload your Image or PDF", type=["png", "jpg", "jpeg", "pdf"])
st.markdown('</div>', unsafe_allow_html=True)

# Extractive summarization using LexRank
def extractive_summary(text, num_sentences=10):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return "\n".join([f"• {sentence}" for sentence in summary])

# PDF text extraction using PyMuPDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in pdf_reader:
        text += page.get_text()
    return text

# Processing logic
if uploaded_file:
    file_name = uploaded_file.name.lower()
    st.success(f"✅ File uploaded: {file_name}")

    if file_name.endswith(".pdf"):
        with st.spinner("🕒 Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)
        st.success("✅ Text extracted from PDF.")
    else:
        try:
            image = Image.open(uploaded_file)
            # No display of uploaded image
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='PNG')
            image_bytes = image_bytes.getvalue()

            with st.spinner("🕒 Extracting text from Image..."):
                result = reader.readtext(image_bytes, detail=0, paragraph=True)
                text = "\n".join(result)
            st.success("✅ Text extracted from Image.")
        except Exception as e:
            st.error(f"⚠ Error while processing image: {e}")

    if text.strip():
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("📜 Extracted Text")
        st.text_area("", text, height=300, key="extracted_text", disabled=True, label_visibility="collapsed")

        if st.button("📋 Generate Fast Summary", key="summarize_button"):
            with st.spinner("🕒 Generating summary..."):
                summary_text = extractive_summary(text, num_sentences=10)
            st.success("✅ Summary Ready!")
            st.subheader("📝 Summary")
            st.text_area("", summary_text, height=300, key="summary_text", disabled=True, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠ No text found to process.")
