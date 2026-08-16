import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()  # Load environment variables from .env file

hf_token = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    or os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)
if hf_token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
    os.environ["HUGGINGFACEHUB_ACCESS_TOKEN"] = hf_token
    os.environ["HF_TOKEN"] = hf_token

#MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

# Export a shared ChatOllama instance for use in multiple scripts.
#llm = ChatOllama(model=MODEL, temperature=0)

def get_llm():
    return ChatOllama(model="qwen2.5:1.5b", temperature=1)

def get_embedding():
    return OllamaEmbeddings(model="mxbai-embed-large", temperature=1)

def get_huggingface_embedding():
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    return embedding

def get_huggingface_llm():
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN or HUGGINGFACEHUB_API_TOKEN or HUGGINGFACEHUB_ACCESS_TOKEN is required for HuggingFaceEndpoint."
        )
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen3-8B",
        task="text-generation",
        huggingfacehub_api_token=hf_token,
    )
    return ChatHuggingFace(llm=llm)
