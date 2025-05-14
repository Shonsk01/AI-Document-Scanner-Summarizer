import streamlit as st
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
import nltk

# Download punkt for summarization
nltk.download('punkt', quiet=True)

# Streamlit UI
st.title("📄 Textify - AI Doc Extractor & Summarizer")
st.caption("Lightweight Version (pytesseract safe for Cloud)")

uploaded_file = st.file_uploader("Upload Image or PDF", type=["png", "jpg", "jpeg", "pdf"])

# Summarization function
def extractive_summary(text, num_sentences=8):
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

# Image OCR using pytesseract
def extract_text_from_image(image_file):
    image = Image.open(image_file).convert("RGB")
    text = pytesseract.image_to_string(image)
    return text

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

    if text.strip():
        st.subheader("📜 Extracted Text")
        st.text_area("Extracted Text", text, height=300)

        if st.button("📋 Fast Summarize"):
            with st.spinner("Generating summary..."):
                summary_text = extractive_summary(text)
            st.success("✅ Summary Ready!")
            st.subheader("Summary")
            st.text_area("Summary", summary_text, height=300)
    else:
        st.warning("⚠ No text found to process.")
