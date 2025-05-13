import streamlit as st
from PIL import Image
import easyocr
import fitz  # PyMuPDF
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
import nltk
import io
import time

# Ensure NLTK resources are present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize OCR reader
reader = easyocr.Reader(['en'])

# Streamlit UI
st.title("📄 Textify -By Shon Sudhir Kamble")

# File uploader
uploaded_file = st.file_uploader("Upload Image or PDF", type=["png", "jpg", "jpeg", "pdf"])

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
    st.write(f"File uploaded: {file_name}")

    if file_name.endswith(".pdf"):
        st.write("Processing PDF...")
        progress_bar = st.progress(0, text="Extracting text from PDF...")
        with st.spinner("Extracting text from PDF..."):
            for percent_complete in range(1, 101, 20):
                time.sleep(0.1)
                progress_bar.progress(percent_complete, text="Extracting text from PDF...")
            text = extract_text_from_pdf(uploaded_file)
        progress_bar.empty()
        st.success("✅ Text extracted from PDF.")
    else:
        st.write("Processing Image...")
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_container_width=True)
            
            # Convert the image to bytes before passing to EasyOCR
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='PNG')
            image_bytes = image_bytes.getvalue()

            progress_bar = st.progress(0, text="Extracting text from Image...")
            with st.spinner("Extracting text from Image..."):
                for percent_complete in range(1, 101, 20):
                    time.sleep(0.1)
                    progress_bar.progress(percent_complete, text="Extracting text from Image...")
                result = reader.readtext(image_bytes, detail=0, paragraph=True)
                text = "\n".join(result)
            progress_bar.empty()
            st.success("✅ Text extracted from Image.")
        except Exception as e:
            st.error(f"Error while processing image: {e}")
            st.write(e)

    # Display extracted text in scrollable area
    if text.strip():
        st.subheader("📜 Extracted Text")
        st.text_area("", text, height=300)

        if st.button("📋 Fast Summarize"):
            progress_bar = st.progress(0, text="Summarizing text...")
            with st.spinner("Generating summary using extractive method..."):
                for percent_complete in range(1, 101, 20):
                    time.sleep(0.1)
                    progress_bar.progress(percent_complete, text="Summarizing text...")
                summary_text = extractive_summary(text, num_sentences=10)
            progress_bar.empty()
            st.success("✅ Summary Ready!")
            st.subheader("Summary")
            st.text_area("", summary_text, height=300)
    else:
        st.warning("⚠ No text found to process.")
