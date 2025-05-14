import streamlit as st
from PIL import Image
import easyocr
import io
import nltk

# Initialize OCR reader
reader = easyocr.Reader(['en'])

# Ensure NLTK resources are present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Streamlit UI
st.title("📄 Textify -by Shon Sudhir Kamble")

# File uploader
uploaded_file = st.file_uploader("Upload Image or PDF", type=["png", "jpg", "jpeg"])

# Extractive summarization using LexRank
def extractive_summary(text, num_sentences=10):
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.nlp.tokenizers import Tokenizer
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return "\n".join([f"• {sentence}" for sentence in summary])

# Processing image logic
def extract_text_from_image(image_file):
    # Open image and convert to RGB (for OCR processing)
    image = Image.open(image_file).convert("RGB")
    
    # Resize the image if too large (keeping aspect ratio)
    base_width = 800
    if image.width > base_width:
        w_percent = (base_width / float(image.width))
        h_size = int((float(image.height) * float(w_percent)))
        image = image.resize((base_width, h_size), Image.LANCZOS)
    
    # Convert the image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()
    
    # Use EasyOCR to extract text
    result = reader.readtext(image_bytes, detail=0, paragraph=True)
    return "\n".join(result)

# Processing logic
if uploaded_file:
    file_name = uploaded_file.name.lower()
    st.write(f"File uploaded: {file_name}")  # Debugging output

    # Check if the image file size is within limits
    if uploaded_file.size > 500_000:  # 500 KB
        st.warning("Image too large! Please upload an image below 500 KB.")
    else:
        st.write("Processing Image...")  # Debugging output
        try:
            text = extract_text_from_image(uploaded_file)
            st.success("✅ Text extracted from Image.")
            
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
        except Exception as e:
            st.error(f"Error while processing image: {e}")
            st.write(e)  # Print the error to the screen for debugging
