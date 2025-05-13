import pytesseract
import streamlit as st
from PIL import Image
from transformers import pipeline
import PyPDF2
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import nltk

# Download the necessary NLTK data (punkt tokenizer)
nltk.download('punkt')

# Sumy extractive summarizer function
def extractive_summary(text, num_sentences=10):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return "\n\n".join([f"• {sentence}" for sentence in summary])

st.title("📄 Textify: PDF & Image Summarizer")

uploaded_file = st.file_uploader("Upload Image or PDF", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file is not None:
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        with st.spinner("Extracting text from PDF..."):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        st.success("✅ Text extracted from PDF.")
    else:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        with st.spinner("Extracting text from Image..."):
            text = pytesseract.image_to_string(image)
        st.success("✅ Text extracted from Image.")

    if text.strip():
        st.subheader("Extracted Text")
        st.text_area("", text, height=300)

        if st.button("📋 Fast Summarize"):
            with st.spinner("Generating summary using extractive method..."):
                summary_text = extractive_summary(text, num_sentences=15)
            st.success("✅ Summary Ready!")
            st.subheader("Summary")
            st.text_area("", summary_text, height=300)
    else:
        st.warning("No text found to process.")
