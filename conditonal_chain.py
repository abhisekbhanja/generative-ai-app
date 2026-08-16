from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableBranch

from model_config import get_llm

llm = get_llm(temperature=0)

class Category(BaseModel):
    category: Literal["technical", "refund"]
    #response:str=Field(description="The category of the customer query")
    #answer: str


parser=StrOutputParser()
parser2=PydanticOutputParser(pydantic_object=Category)

query_promt=ChatPromptTemplate.from_messages([
    ("system", "You are a customer query analyst"),
    ("user", "classify the query wheather it is technical or a refund request: \n {question} \n {format_instructions}")
    ]).partial(format_instructions=parser2.get_format_instructions())

technical_promt=ChatPromptTemplate.from_messages([
    ("system", "You are a technical analyst"),
    ("user", "send a messege to the technical team: \n {answer}")
    ])

financial_promt=ChatPromptTemplate.from_messages([
    ("system", "You are a financial analyst"),
    ("user", "send a messege to the financial team: \n {answer}")
    ])

default_promt=ChatPromptTemplate.from_messages([
    # ("system", "You are a financial analyst"),
    ("user", "send a messege to create a ticket: \n {answer}")
    ])

technical_chain=technical_promt | llm | parser
financial_chain=financial_promt | llm | parser
default_chain=default_promt | llm | parser

runnnable=RunnableBranch(
    (lambda x: x.category=="technical", technical_chain),
    (lambda x: x.category=="refund", financial_chain),
    default_chain
)
chain=query_promt | llm | parser2 | runnnable | parser
output=chain.invoke({"question": "I'm unable to connect to the server. It keeps showing \"Connection Refused\"."})
print(output)
#print(chain.get_graph().draw_ascii())

