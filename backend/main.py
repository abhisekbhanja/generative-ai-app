from fastapi import FastAPI, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from upload_service import process_uploaded_file
from llm_service import llmcall

app = FastAPI(title="Simple FastAPI App")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    print("Hello from FastAPI!")
    return {"message": "Hello from FastAPI!"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    message = process_uploaded_file(file)
    print(message)
    return {"message": message, "status_code": status.HTTP_201_CREATED}

class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_question(request: AskRequest):
    answer = llmcall(request.question)
    print("Answer:", answer)
    return answer