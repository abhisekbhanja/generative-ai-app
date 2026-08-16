from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from model_config import get_embedding, get_huggingface_embedding, get_huggingface_llm, get_llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
#pip install -r requirements.txt

embedding = get_embedding()
#get data from vectorstore
db = FAISS.load_local(
    "sales_index",
    embedding,
    allow_dangerous_deserialization=True
)

#query = "What is the leave policy for employees?"

#retrieve relevant documents
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
#retrieved_docs = retriever.invoke(query)

def get_context_text(docs):
    return  "\n\n".join(doc.page_content for doc in docs)
#print(context_text)

llm = get_llm()
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Use the following context to answer the question.\n\nContext:\n{context}\n\nquery:\n{query}\n\nAnswer:"
)
# formatted_prompt = prompt.format(context=get_context_text(), query="explain the leave policy")
# result = llm.invoke(formatted_prompt)
# print(" ")
# print("Answer:", result.content)

paralel_chain = RunnableParallel(
    context=retriever | RunnableLambda(lambda docs: get_context_text(docs)),
    query=RunnablePassthrough() 
)

# parallel_result = paralel_chain.invoke(query)

parser=StrOutputParser()
final_chain = paralel_chain |prompt | llm | parser
final_result = final_chain.invoke("how many people ordered in february?")
print("Answer:", final_result)

