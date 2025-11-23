from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

# Get API token from Streamlit secrets or environment variable
try:
    api_token = st.secrets["HUGGINGFACEHUB_ACCESS_TOKEN"]
except (KeyError, FileNotFoundError, AttributeError):
    api_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

if not api_token:
    st.error("⚠️ HuggingFace API token not found! Please add it to Streamlit secrets or set the environment variable.")
    st.info("Go to App Settings → Secrets and add: HUGGINGFACEHUB_ACCESS_TOKEN = 'your_token_here'")
    st.stop()

# Initialize HuggingFace endpoint - using a model that works reliably with HF Inference API
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token=api_token,
    temperature=0.7,
    max_new_tokens=512,
)

model = ChatHuggingFace(llm=llm)

st.header("Chat with Mistral 7B Instruct Model")
user_input = st.text_input("Enter your prompt:")

if st.button("Generate Response"):
    if not user_input:
        st.warning("Please enter a prompt first!")
    else:
        with st.spinner("Generating response..."):
            try:
                result = model.invoke(user_input)
                st.write(result.content)
            except Exception as e:
                st.error(f"Error: {str(e)}")