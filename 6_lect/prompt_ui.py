from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
    # max_new_tokens=512,
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)

st.header("Chat with Llama 3.3-70B-Instruct Model")
user_input = st.text_input("Enter your prompt:")

if st.button("Generate Response"):
    result = model.invoke(user_input)
    st.write(result.content)