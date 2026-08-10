import io
import os
import socket
import logging
import boto3
from botocore.client import Config
import pypdf
import docx

from app.core.config import settings

logger = logging.getLogger(__name__)

def get_minio_endpoint() -> str:
    """
    Get the MinIO endpoint, automatically fallback to localhost if the docker service name 'minio' is not resolvable.
    """
    endpoint = settings.MINIO_ENDPOINT
    host = endpoint.split(":")[0]
    try:
        socket.gethostbyname(host)
    except socket.gaierror:
        # Fallback to localhost if 'minio' can't be resolved (running on host machine)
        endpoint = endpoint.replace("minio", "localhost")
    return endpoint

def get_s3_client():
    endpoint = get_minio_endpoint()
    schema = "https" if settings.MINIO_SECURE else "http"
    return boto3.client(
        "s3",
        endpoint_url=f"{schema}://{endpoint}",
        aws_access_key_id=settings.MINIO_ACCESS_KEY,
        aws_secret_access_key=settings.MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )

class FileProcessor:
    def __init__(self):
        self.s3_client = get_s3_client()
        self.bucket_name = settings.MINIO_BUCKET

    def download_file(self, file_path: str) -> bytes:
        """
        Download file content as bytes from MinIO.
        """
        logger.info(f"Downloading file from MinIO: bucket={self.bucket_name}, key={file_path}")
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_path)
        return response["Body"].read()

    def process_file(self, file_path: str) -> dict:
        """
        Download the file, extract text, and compute page and word counts.
        """
        file_bytes = self.download_file(file_path)
        ext = os.path.splitext(file_path.lower())[1]

        text = ""
        page_count = 1

        if ext == ".pdf":
            text, page_count = self._extract_pdf(file_bytes)
        elif ext == ".docx":
            text, page_count = self._extract_docx(file_bytes)
        elif ext in (".txt", ".text"):
            text, page_count = self._extract_txt(file_bytes)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        word_count = len(text.split())

        return {
            "text": text,
            "page_count": page_count,
            "word_count": word_count,
        }

    def _extract_pdf(self, file_bytes: bytes) -> tuple[str, int]:
        pdf_file = io.BytesIO(file_bytes)
        reader = pypdf.PdfReader(pdf_file)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n".join(text_parts), len(reader.pages)

    def _extract_docx(self, file_bytes: bytes) -> tuple[str, int]:
        docx_file = io.BytesIO(file_bytes)
        doc = docx.Document(docx_file)
        text_parts = [p.text for p in doc.paragraphs]
        return "\n".join(text_parts), 1  # DOCX files do not store a static page count in flow layout

    def _extract_txt(self, file_bytes: bytes) -> tuple[str, int]:
        text = file_bytes.decode("utf-8", errors="ignore")
        return text, 1
    def upload_file(self, file_path: str, content: bytes) -> bool:
        try:
            self.s3_client.put_object(Bucket=self.bucket_name, Key=file_path, Body=content)
            return True
        except Exception as e:
            raise
