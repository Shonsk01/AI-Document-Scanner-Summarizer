import streamlit as st
import easyocr
import PyPDF2
from summa import summarizer
from PIL import Image
import pytesseract
import nltk
from nltk.tokenize import sent_tokenize
import pdfplumber
import io

# Initialize the OCR reader
reader = easyocr.Reader(['en'])

# Function to process text for summarization
def extractive_summary(text, num_sentences=5):
    """
    Extractive text summarization using the Summa library
    """
    return summarizer.summarize(text, num_sentences=num_sentences)

# OCR processing for images
def ocr_from_image(image):
    """
    Extract text from an image using EasyOCR
    """
    result = reader.readtext(image)
    text = " ".join([res[1] for res in result])
    return text

# OCR processing for PDFs
def ocr_from_pdf(file):
    """
    Extract text from PDF using PyPDF2 and pdfplumber
    """
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to handle different file types
def process_file(uploaded_file):
    """
    Determine the type of the uploaded file and extract text accordingly
    """
    file_type = uploaded_file.type
    text = ""

    if "image" in file_type:
        # Process as an image file
        image = Image.open(uploaded_file)
        text = ocr_from_image(image)
    elif "pdf" in file_type:
        # Process as a PDF file
        text = ocr_from_pdf(uploaded_file)
    elif "text" in file_type:
        # Process as a plain text file
        text = str(uploaded_file.read(), "utf-8")
    
    return text

# Streamlit UI setup
st.title("Document Text Extractor & Summarizer")
st.write("Upload your image, PDF, or text file to extract and summarize text.")

# File uploader
uploaded_file = st.file_uploader("Choose a file (image, PDF, or text)", type=["jpg", "jpeg", "png", "pdf", "txt"])

if uploaded_file is not None:
    # Extract text based on file type
    text = process_file(uploaded_file)

    if text:
        st.subheader("Extracted Text")
        st.write(text)

        # Tokenize sentences for summarization
        nltk.download('punkt')
        sentences = sent_tokenize(text)

        # Summarize the text
        if len(sentences) > 5:
            summary = extractive_summary(text, num_sentences=5)
            st.subheader("Summary")
            st.write(summary)
        else:
            st.subheader("Summary")
            st.write("Text is too short for summarization.")
    else:
        st.write("No text extracted from the file.")
