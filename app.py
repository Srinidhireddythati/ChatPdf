import streamlit as st
import time
import PyPDF2
import faiss
from sentence_transformers import SentenceTransformer

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def embed_text(text, model):
    sentences = text.split('.')
    embeddings = model.encode(sentences)
    return sentences, embeddings

def setup_vector_store(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_similar_texts(query, index, sentences, model, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [sentences[i] for i in indices[0]]

def stream_data(text, delay=0.02):
    for word in text.split():
        yield word + " "
        time.sleep(delay)

st.title("PDF Q&A")

# Load Sentence Transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    sentences, embeddings = embed_text(pdf_text, embedding_model)
    vector_index = setup_vector_store(embeddings)
    st.success("PDF processed and vector store set up.")

# Input prompt
prompt = st.text_input("Ask a question...")

if prompt and uploaded_file:
    # Display input prompt from user
    with st.spinner("Thinking..."):
        try:
            relevant_texts = retrieve_similar_texts(prompt, vector_index, sentences, embedding_model)
            response = " ".join(relevant_texts)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            response = ""

        # Display streamed response
        if response:
            with st.empty():
                streamed_response = ""
                for word in stream_data(response):
                    streamed_response += word
                    st.markdown(streamed_response)
