import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

DOCLING_URL = os.getenv("DOCLING_URL", "http://localhost:5002")

def process_document_with_docling(file_path):
    """
    Process a document using the Docling Server.
    """
    try:
        with open(file_path, 'rb') as file:
            files = {'files': file}
            response = requests.post(f"{DOCLING_URL}/v1alpha/convert/file/async", files=files)
        
        if response.status_code == 200:
            task_id = response.json().get('task_id')
            if not task_id:
                raise Exception("No task ID returned from Docling server.")
            # Poll for the task result
            for _ in range(10):  # Poll up to 10 times
                poll = requests.get(f"{DOCLING_URL}/v1alpha/status/poll/{task_id}", params={'wait': 1})
                if poll.status_code == 200:
                    status = poll.json().get("task_status")
                    if status == "success":
                        # Now fetch the result
                        result = requests.get(f"{DOCLING_URL}/v1alpha/result/{task_id}")
                        if result.status_code == 200:
                            return result.json()
                        else:
                            raise Exception(f"Error fetching result: {result.status_code} - {result.text}")
                    elif status in ("failure", "partial_success"):
                        raise Exception(f"Docling task failed: {poll.json()}")
                    # else: still pending, wait and retry
                time.sleep(1)
            raise Exception("Docling task did not complete in time.")
        else:
            raise Exception(f"Error processing document: {response.status_code} - {response.text}")
    except Exception as e:
        raise Exception(f"Failed to process document {file_path}: {str(e)}")

def get_docling_health():
    """
    Get the health status of the Docling Server.
    """
    try:
        response = requests.get(f"{DOCLING_URL}/health")
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error getting health status: {response.status_code} - {response.text}")
    except Exception as e:
        raise Exception(f"Failed to get Docling health: {str(e)}")

if __name__ == "__main__":
    test_file_path = "Ben West CV.pdf"  # Replace with your test file path
    # Create a dummy test file for demonstration
    if not os.path.exists(test_file_path):
        with open(test_file_path, 'w') as f:
            f.write("This is a test document for Docling processing.")

    processed_text = process_document_with_docling(test_file_path)
    if processed_text:
        print("Document processed successfully:")
        print(processed_text)
    else:
        print("Failed to extract text with Docling.")