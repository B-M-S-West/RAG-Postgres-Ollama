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

def process_document_with_docling_from_url(file_url):
    """
    Process a document using the Docling Server.
    """
    try:
        request_body = {
            'http_sources': [
                {
                    'url': file_url
                }
            ],
            'options': {
                'to_formats': ['md'],
                'do_ocr': True,
                'document_timeout': 300.0,  # 5 minutes timeout
                'abort_on_error': False
            }
        }
        
        response = requests.post(
            f"{DOCLING_URL}/v1alpha/convert/source/async", 
            json=request_body
        )
        
        if response.status_code == 200:
            task_id = response.json().get('task_id')
            if not task_id:
                raise Exception("No task ID returned from Docling server.")
            
            print(f"Task created with ID: {task_id}")
            
            # Poll for the task result
            for attempt in range(30):
                poll = requests.get(
                    f"{DOCLING_URL}/v1alpha/status/poll/{task_id}", 
                    params={'wait': 2}
                )
                
                if poll.status_code == 200:
                    poll_data = poll.json()
                    status = poll_data.get("task_status")
                    
                    print(f"Attempt {attempt + 1}: Status = {status}")
                    
                    # Print task meta information if available
                    task_meta = poll_data.get('task_meta')
                    if task_meta:
                        print(f"Task meta: {task_meta}")
                    
                    if status == "success":
                        # Fetch the result
                        result = requests.get(f"{DOCLING_URL}/v1alpha/result/{task_id}")
                        
                        if result.status_code == 200:
                            result_data = result.json()
                            document = result_data.get('document', {})
                            
                            if document.get('md_content'):
                                return document['md_content']
                            elif document.get('text_content'):
                                return document['text_content']
                            elif document.get('html_content'):
                                return document['html_content']
                            elif document.get('json_content'):
                                return str(document['json_content'])
                            else:
                                return str(result_data)
                        else:
                            raise Exception(f"Error fetching result: {result.status_code} - {result.text}")
                    
                    elif status == "failure":
                        # For failed tasks, we can't get the result, but we can get more info from the poll response
                        error_msg = f"Docling task failed. Task ID: {task_id}"
                        
                        # Check if there's any additional info in the poll response
                        if 'task_meta' in poll_data and poll_data['task_meta']:
                            error_msg += f". Task meta: {poll_data['task_meta']}"
                        
                        # The actual error details might not be available via API
                        # This suggests a server-side issue
                        error_msg += ". This typically indicates: 1) URL not accessible from Docling server, 2) Unsupported file format, 3) File corruption, or 4) Server configuration issue."
                        
                        raise Exception(error_msg)
                    
                    elif status == "partial_success":
                        # Try to get the result even for partial success
                        try:
                            result = requests.get(f"{DOCLING_URL}/v1alpha/result/{task_id}")
                            if result.status_code == 200:
                                result_data = result.json()
                                document = result_data.get('document', {})
                                
                                # Return whatever content we got
                                if document.get('md_content'):
                                    return document['md_content']
                                elif document.get('text_content'):
                                    return document['text_content']
                                else:
                                    return str(result_data)
                            else:
                                raise Exception(f"Partial success but could not fetch result: {result.status_code}")
                        except Exception as e:
                            raise Exception(f"Task completed with partial success but error fetching result: {str(e)}")
                    
                    # Status is "pending" or "started", continue polling
                else:
                    raise Exception(f"Error polling task status: {poll.status_code} - {poll.text}")
                
                time.sleep(3)  # Wait a bit longer between polls
            
            raise Exception("Docling task did not complete in time.")
        else:
            raise Exception(f"Error creating task: {response.status_code} - {response.text}")
            
    except Exception as e:
        raise Exception(f"Failed to process document {file_url}: {str(e)}")
    
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