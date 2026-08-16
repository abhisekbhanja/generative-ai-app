import os
from langchain_ollama import ChatOllama
#from langchain_core.prompts import ChatPromptTemplate,PromtTemplate
from langchain_core.output_parsers import StrOutputParser

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
llm = ChatOllama(model=MODEL, temperature=0)

ans = llm.invoke("what is 268686+87868792")

print(ans.content)



