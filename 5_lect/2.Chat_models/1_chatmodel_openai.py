from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

# model = ChatOpenAI(model="gpt-3.5-turbo")
# via OpenRouter way
model = ChatOpenAI(
    # model="google/gemini-2.0-flash-exp:free",
    model="openai/gpt-4o-mini",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)
result= model.invoke("What is the capital of India?")
print(result.content)