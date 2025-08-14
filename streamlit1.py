import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from operator import itemgetter
# import llama_3

st.title("WELCOME")
st.header("Ask a Question:")
input1=st.text_input(label="Enter here")
st.sidebar.title("Select a pdf")
file=st.sidebar.file_uploader("choose")
# st.write(file)

