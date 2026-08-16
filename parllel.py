from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

from model_config import get_llm

llm = get_llm(temperature=1)

parser = StrOutputParser()

# Sentiment
sentiment_prompt = ChatPromptTemplate.from_template(
    "Tell whether the sentiment is Positive, Negative, or Neutral.\n\n{text}"
)

# Summary
summary_prompt = ChatPromptTemplate.from_template(
    "Summarize the following review in one sentence.\n\n{text}"
)

# Category
category_prompt = ChatPromptTemplate.from_template(
    "Categorize this review as Product, Delivery, or Payment.\n\n{text}"
)

parallel_chain = RunnableParallel(
    sentiment=sentiment_prompt | llm | parser,
    summary=summary_prompt | llm | parser,
    category=category_prompt | llm | parser,
)

result = parallel_chain.invoke(
    {
        "text": "The product quality is excellent but delivery was very slow."
    }
)

print(result)