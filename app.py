import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

restaurant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a creative naming assistant. Return only one short restaurant name inspired by the given country.",
        ),
        ("human", "Country: {country}"),
    ]
)

recipe_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a food expert. Give 3 simple recipe ideas in bullet points based on the country and restaurant name.",
        ),
        ("human", "Country: {country}\nRestaurant name: {restaurant_name}"),
    ]
)

llm = ChatOllama(model=MODEL, temperature=0)
restaurant_chain = restaurant_prompt | llm | StrOutputParser()
recipe_chain = recipe_prompt | llm | StrOutputParser()


def ask_ai(country: str) -> str:
    if not country.strip():
        return "Please enter a country name or type 'exit' to quit."

    if country.strip().lower() in {"exit", "quit", "bye"}:
        return "Goodbye!"

    try:
        restaurant_name = restaurant_chain.invoke({"country": country}).strip()
        recipes = recipe_chain.invoke({"country": country, "restaurant_name": restaurant_name}).strip()
        return f"Restaurant name: {restaurant_name}\n\nRecipes:\n{recipes}"
    except Exception as exc:
        return f"Could not reach Ollama. Make sure Ollama is running and the model '{MODEL}' is installed. Error: {exc}"


if __name__ == "__main__":
    print("Country-based restaurant and recipe generator. Type 'exit' to quit.")

    while True:
        try:
            country = input("Country: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break

        if not country:
            continue

        reply = ask_ai(country)
        print(f"\n{reply}\n")

        if reply == "Goodbye!":
            break
