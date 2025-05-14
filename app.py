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

# Streamlit UI
st.title("📄 Textify - by Shon Sudhir Kamble")

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

# Extract text from image using OCR
def extract_text_from_image(image):
    # Check image size before processing (limit to small images only)
    max_size = 2 * 1024 * 1024  # 2MB max
    if len(image.getdata()) > max_size:
        raise ValueError("The uploaded image is too large. Please upload a smaller image.")

    # Process the image with EasyOCR
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()
    
    # Perform OCR
    result = reader.readtext(image_bytes, detail=0, paragraph=True)
    return "\n".join(result)

# Processing logic
if uploaded_file:
    file_name = uploaded_file.name.lower()
    st.write(f"File uploaded: {file_name}")  # Debugging output

    if file_name.endswith(".pdf"):
        st.write("Processing PDF...")  # Debugging output
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)
        st.success("✅ Text extracted from PDF.")
    else:
        st.write("Processing Image...")  # Debugging output
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_container_width=True)

            # Extract text from image
            with st.spinner("Extracting text from Image..."):
                text = extract_text_from_image(image)
            st.success("✅ Text extracted from Image.")
        except Exception as e:
            st.error(f"Error while processing image: {e}")
            st.write(e)  # Print the error to the screen for debugging

    # Display extracted text in scrollable area
    if text.strip():
        st.subheader("📜 Extracted Text")
        st.text_area("", text, height=300)

        if st.button("📋 Fast Summarize"):
            with st.spinner("Generating summary using extractive method..."):
                summary_text = extractive_summary(text, num_sentences=10)
            st.success("✅ Summary Ready!")
            st.subheader("Summary")
            st.text_area("", summary_text, height=300)
    else:
        st.warning("⚠ No text found to process.")
