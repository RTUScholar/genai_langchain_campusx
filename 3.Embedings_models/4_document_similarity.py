# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()


# emedings = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)
query = "Delhi is the capital of which country?"

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France",
    "mumbai is the capital of maharashtra",
    "berlin is the capital of germany",
    "canberra is the capital of australia",
    "ottawa is the capital of canada"
]

doc_embeddings = embeddings.embed_documents(documents)
single_embedings = embeddings.embed_query(query)

score = cosine_similarity([single_embedings], doc_embeddings)

print(str(score))
print("query is: ",query)

result = sorted(list(enumerate(score[0])), key=lambda x:x[1])[-1]
print(str(result))
