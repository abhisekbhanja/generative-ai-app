from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from model_config import get_embedding, get_huggingface_embedding, get_huggingface_llm

#load documents
text = Path("test.txt").read_text(encoding="utf-8")
documents = [Document(page_content=text, metadata={"source": "test.txt"})]
print("Loaded documents:", len(documents))

#split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=75, chunk_overlap=5)
chunks = text_splitter.split_documents(documents)
#print(chunks)
print("Split into chunks:", len(chunks))


print("Creating vectorstore... please wait")
embedding = get_embedding()
db = FAISS.from_documents(chunks, embedding)
print("Vectorstore created successfully.")
# Save vector database
db.save_local("faiss_index1")
print("Vectorstore saved locally as 'faiss_index1'.")

