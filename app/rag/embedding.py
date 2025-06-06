import os
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")

def get_embedding(text):
    """
    Sends text to Ollama embedding model and returns the embedding vector.
    """
    url = f"{OLLAMA_BASE_URL}/api/embed"

    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("embeddings")
    except requests.RequestException as e:
        print(f"Error fetching embedding: {e}")
        return None
    
if __name__ == "__main__":
    # Example usage
    sample_text = "This is a sample text for embedding."
    embedding = get_embedding(sample_text)
    print(f"Embedding length: {len(embedding)}")
    print(embedding[:5])  # Print first 5 values for a quick check