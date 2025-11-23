from langchain_openai  import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()

# embedings = OpenAIEmbeddings(model="text-embedding-3-large",
#                             dimensions=32)


# using HF endpoint
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

vector = embeddings.embed_query("What is the capital of India?")
print(vector)