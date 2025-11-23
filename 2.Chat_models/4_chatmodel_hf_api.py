from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    # repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
    # max_new_tokens=512,
    temperature=0.7,
)
model = ChatHuggingFace(llm=llm)

result = model.invoke("can you give me a brief overview of the history of artificial intelligence?")
print(result.content)