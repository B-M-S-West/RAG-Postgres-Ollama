import boto3
import os

def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=os.getenv('MINIO_ENDPOINT'),  # Use this for MinIO
        aws_access_key_id=os.getenv('MINIO_ACCESS_KEY'),
        aws_secret_access_key=os.getenv('MINIO_SECRET_KEY'),
        region_name="",
        use_ssl=os.getenv("MINIO_SECURE", "false").lower() == "true"
    )

def upload_file_to_s3(file_obj, filename, bucket=None):
    bucket = bucket or os.getenv("MINIO_BUCKET")
    s3 = get_s3_client()
    s3.upload_fileobj(file_obj, bucket, filename)
    endpoint = os.getenv("DOCKER_MINIO_ENDPOINT")
    return f"{endpoint}/{bucket}/{filename}"