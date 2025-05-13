import streamlit as st
from PIL import Image
import easyocr
import fitz  # PyMuPDF
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
import nltk
import io
import os
import time

# ---- Safe NLTK setup for Streamlit Cloud ----
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_dir)
os.makedirs(nltk_data_dir, exist_ok=True)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_dir)

# ---- Initialize OCR reader ----
reader = easyocr.Reader(['en'])

# ---- UI ----
st.title("📄 AI Document Scanner & Summarizer (Image & PDF) by Shon")

uploaded_file = st.file_uploader("Upload Image or PDF", type=["png", "jpg", "jpeg", "pdf"])

# ---- Functions ----
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

# ---- Processing ----
if uploaded_file:
    file_name = uploaded_file.name.lower()
    st.write(f"📂 File uploaded: `{file_name}`")

    progress_bar = st.progress(0)

    if file_name.endswith(".pdf"):
        with st.spinner("Extracting text from PDF..."):
            progress_bar.progress(30)
            text = extract_text_from_pdf(uploaded_file)
            time.sleep(0.5)
            progress_bar.progress(100)
        st.success("✅ Text extracted from PDF.")
    else:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_container_width=True)
            progress_bar.progress(30)

            image_bytes = io.BytesIO()
            image.save(image_bytes, format='PNG')
            image_bytes = image_bytes.getvalue()

            with st.spinner("Extracting text from Image..."):
                result = reader.readtext(image_bytes, detail=0, paragraph=True)
                time.sleep(0.5)
                progress_bar.progress(100)
                text = "\n".join(result)
            st.success("✅ Text extracted from Image.")
        except Exception as e:
            st.error(f"❌ Error while processing image: {e}")
            st.write(e)

    # ---- Show Extracted text ----
    if text.strip():
        st.subheader("📜 Extracted Text")
        st.text_area("", text, height=300)

        if st.button("📋 Fast Summarize"):
            progress_bar2 = st.progress(0)
            with st.spinner("Generating summary..."):
                progress_bar2.progress(30)
                summary_text = extractive_summary(text, num_sentences=10)
                time.sleep(0.5)
                progress_bar2.progress(100)
            st.success("✅ Summary Ready!")
            st.subheader("Summary")
            st.text_area("", summary_text, height=300)
    else:
        st.warning("⚠ No text found to process.")
