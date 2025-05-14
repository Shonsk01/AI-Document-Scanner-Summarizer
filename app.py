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

# Initialize OCR reader
reader = easyocr.Reader(['en'])

# ------------------- UI -------------------
st.set_page_config(page_title="Textify AI", page_icon="📄", layout="wide")
st.title("📄 Textify - by Shon Sudhir Kamble")
st.markdown("### Extract text & summarize from PDFs or Images using AI 🧠")

# Upload section
st.markdown("---")
uploaded_file = st.file_uploader("📤 Upload an **Image** or **PDF** file", type=["png", "jpg", "jpeg", "pdf"])

# ------------------- Functions -------------------
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

# ------------------- Processing -------------------
if uploaded_file:
    file_name = uploaded_file.name.lower()
    st.info(f"📁 **File Uploaded:** {file_name}")

    # PDF Handling
    if file_name.endswith(".pdf"):
        st.markdown("### 📑 PDF Processing")
        with st.spinner("🔍 Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)
        st.success("✅ Text extracted from PDF.")

    # Image Handling
    else:
        st.markdown("### 🖼️ Image Processing")
        try:
            image = Image.open(uploaded_file)
            with st.container():
                st.image(image, caption='Uploaded Image', use_container_width=True, channels="RGB")

            # Convert image to bytes
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='PNG')
            image_bytes = image_bytes.getvalue()

            with st.spinner("🔍 Extracting text from Image..."):
                result = reader.readtext(image_bytes, detail=0, paragraph=True)
                text = "\n".join(result)
            st.success("✅ Text extracted from Image.")
        except Exception as e:
            st.error(f"❌ Error while processing image: {e}")
            st.write(e)

    # ------------------- Display Extracted Text -------------------
    if text.strip():
        st.markdown("### 📜 Extracted Text")
        st.text_area("Extracted Text", text, height=300)

        # Summarization button
        if st.button("📋 Fast Summarize"):
            with st.spinner("💡 Generating summary..."):
                summary_text = extractive_summary(text, num_sentences=10)
            st.success("✅ Summary Ready!")
            st.markdown("### 📝 Summary")
            st.text_area("Summary", summary_text, height=300)
    else:
        st.warning("⚠ No text found to process.")
