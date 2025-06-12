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

st.header("Query the Knowledge Base")
query = st.text_input("Enter your query")

if st.button("Submit Query"):
    params = {"q": query}
    with st.spinner("Searching..."):
        response = requests.get(f"{BACKEND_URL}/query", params=params)
        if response.status_code == 200:
            results = response.json()["results"]
            st.subheader("Query results:")
            if results:
                for result in results:
                    st.write(f"**Document ID:** {result['id']}")
                    st.write(f"**Content:** {result['content']}")
            else:
                st.write("No results found.")
        else:
            st.error(f"Error: {response.text}")

