import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
llm = ChatOllama(model=MODEL, temperature=0)

restaurant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a creative naming assistant. Return only one short restaurant name inspired by the given country.",
        ),
        ("human", "Country: {country}"),
    ])

recipe_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a food expert. Give 3 simple recipe ideas in bullet points based on the country and restaurant name.",
        ),
        ("human", "Country: {country}\nRestaurant name: {restaurant_name}"),
    ])

result_chain = restaurant_prompt | llm | StrOutputParser()
recipe_chain = recipe_prompt | llm | StrOutputParser()

print("Loading...")
restaurant_result= result_chain.invoke({"country": "England"})
result=recipe_chain.invoke({"country": "England", "restaurant_name": restaurant_result})
print(result)



