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
nltk.download('punkt', quiet=True)

# Initialize OCR reader
reader = easyocr.Reader(['en'])

# Streamlit page settings
st.set_page_config(page_title="Textify - AI Document Extractor & Summarizer", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
        .main {background-color: #f8f9fa;}
        .title {font-size: 36px; color: #1c1c1c; font-weight: bold;}
        .subtitle {font-size: 20px; color: #007bff;}
        .upload-section {padding: 20px; background-color: #ffffff; border-radius: 10px; box-shadow: 0px 4px 10px rgba(0,0,0,0.05);}
        .summary-section {padding: 20px; background-color: #f8f9fa; border-radius: 10px; box-shadow: 0px 4px 10px rgba(0,0,0,0.05);}
        .custom-textarea {padding: 20px; background-color: #ffffff; color: #000000; border-radius: 10px; box-shadow: 0px 4px 10px rgba(0,0,0,0.05); font-size: 16px; line-height: 1.6; overflow-y: auto; max-height: 400px;}
        .small-image {border-radius: 10px; box-shadow: 0px 4px 10px rgba(0,0,0,0.05); width: 200px; height: auto; margin-top: 20px;}
    </style>
    """, unsafe_allow_html=True)

# Title and intro
st.markdown('<h1 class="title">📄 Textify - by Shon Sudhir Kamble</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered Document Extractor & Summarizer</p>', unsafe_allow_html=True)

# File uploader section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Image or PDF", type=["png", "jpg", "jpeg", "pdf"])
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

# Process uploaded file
if uploaded_file:
    file_name = uploaded_file.name.lower()
    st.write(f"**File uploaded:** {file_name}")

    text = ""

    if file_name.endswith(".pdf"):
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)
        st.success("✅ Text extracted from PDF.")
    else:
        try:
            image = Image.open(uploaded_file)
            # Show small preview only
            st.image(image, caption='Uploaded Image', width=200)

            # Convert image to bytes for OCR
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='PNG')
            image_bytes = image_bytes.getvalue()

            with st.spinner("Extracting text from Image..."):
                result = reader.readtext(image_bytes, detail=0, paragraph=True)
                text = "\n".join(result)
            st.success("✅ Text extracted from Image.")
        except Exception as e:
            st.error(f"Error processing image: {e}")

    # Display extracted text
    if text.strip():
        st.markdown('<div class="summary-section">', unsafe_allow_html=True)
        st.subheader("📜 Extracted Text")

        safe_text = text.replace("\n", "<br>")
        st.markdown(f"""<div class="custom-textarea">{safe_text}</div>""", unsafe_allow_html=True)

        # Summarize button
        if st.button("📋 Fast Summarize", help="Generate summary from extracted text"):
            with st.spinner("Generating summary..."):
                summary_text = extractive_summary(text, num_sentences=10)
            st.success("✅ Summary Ready!")

            st.subheader("📝 Summary")
            safe_summary = summary_text.replace("\n", "<br>")
            st.markdown(f"""<div class="custom-textarea">{safe_summary}</div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠ No text found to process.")
