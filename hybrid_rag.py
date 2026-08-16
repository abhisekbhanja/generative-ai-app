from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.retrievers import BM25Retriever
#from langchain.retrievers import EnsembleRetriever
from langchain_classic.retrievers import EnsembleRetriever
from model_config import get_embedding, get_llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
#pip install -r requirements.txt
#pip install rank_bm25   # required for BM25Retriever

embedding = get_embedding()

# get data from vectorstore
db = FAISS.load_local(
    "faiss_index1",
    embedding,
    allow_dangerous_deserialization=True
)

query = "What is the leave policy for employees?"

# --- Dense retriever (existing FAISS) ---
dense_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# --- Sparse retriever (BM25) ---
# BM25 needs the raw documents, not just the FAISS index.
# Pull them out of the FAISS docstore so we don't need a separate loader.
all_docs = list(db.docstore._dict.values())

bm25_retriever = BM25Retriever.from_documents(all_docs)
bm25_retriever.k = 3

# --- Hybrid retriever: combine dense + sparse ---
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.4, 0.6]  # tune: more weight toward dense embeddings, adjust as needed
)

def get_context_text(docs):
    return "\n\n".join(doc.page_content for doc in docs)


llm = get_llm()
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Use the following context to answer the question.\n\nContext:\n{context}\n\nquery:\n{query}\n\nAnswer:"
)

parallel_chain = RunnableParallel(
    context=hybrid_retriever | RunnableLambda(lambda docs: get_context_text(docs)),
    query=RunnablePassthrough()
)

parser = StrOutputParser()
final_chain = parallel_chain | prompt | llm | parser
final_result = final_chain.invoke(query)
print(" ")
print("Answer:", final_result)