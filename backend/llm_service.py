from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from model_config import get_embedding, get_llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

index_path = Path("faiss_index")

embedding = get_embedding()
#get data from vectorstore
db = FAISS.load_local(
    index_path,
    embedding,
    allow_dangerous_deserialization=True
)

#query = "What is the leave policy for employees?"

#retrieve relevant documents
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})


def get_context_text(docs):
    return  "\n\n".join(doc.page_content for doc in docs)

llm = get_llm()
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Use the following context to answer the question.\n\nContext:\n{context}\n\nquery:\n{query}\n\nAnswer:"
)

paralel_chain = RunnableParallel(
    context=retriever | RunnableLambda(lambda docs: get_context_text(docs)),
    query=RunnablePassthrough() 
)

# parallel_result = paralel_chain.invoke(query)

parser=StrOutputParser()
final_chain = paralel_chain |prompt | llm | parser

def llmcall(query):
    final_result = final_chain.invoke(query)
    return final_result


