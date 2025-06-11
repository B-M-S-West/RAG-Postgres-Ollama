import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.title("RAG Pipeline Demo")

st.header("Upload and Process Document")
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "pptx", "docx"])

if uploaded_file is not None:
    st.write(f"Selected file: {uploaded_file.name}")
    if st.button("Process Document"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        with st.spinner("Processing..."):
            response = requests.post(f"{BACKEND_URL}/process", files=files)
            if response.status_code == 200:
                st.success("Document processed successfully!")
                st.json(response.json())
            else:
                st.error(f"Error: {response.text}")