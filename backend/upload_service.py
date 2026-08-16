import logging
from pathlib import Path

from fastapi import UploadFile
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model_config import get_embedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_uploaded_file(file: UploadFile) -> str:
    contents = file.file.read().decode("utf-8")
    documents = [Document(page_content=contents, metadata={"source": file.filename})]

    logger.info("Loaded documents: %s", len(documents))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=75, chunk_overlap=5)
    chunks = text_splitter.split_documents(documents)
    logger.info("Split into chunks: %s", len(chunks))

    logger.info("Creating vectorstore... please wait")
    embedding = get_embedding()
    db = FAISS.from_documents(chunks, embedding)
    logger.info("Vectorstore created successfully.")

    index_path = Path("faiss_index")
    db.save_local(str(index_path))
    logger.info("Vectorstore saved locally as 'faiss_index'.")

    return f"File uploaded and processed: {file.filename}"
