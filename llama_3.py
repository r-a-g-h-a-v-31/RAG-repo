pdffile="The Ultimate Python Handbook.pdf"
Model="llama3.1"

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from operator import itemgetter

loader=PyPDFLoader(pdffile)
pages=loader.load()

splitter=RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=200)
chunks=splitter.split_documents(pages)

embeddings=OllamaEmbeddings(model=Model)
vectorstore=DocArrayInMemorySearch.from_documents(chunks,embeddings)
retriever=vectorstore.as_retriever()
# retriever.invoke("What can you get away with when you only have a small number of users?")

MODEL=ChatOllama(model=Model)

parser=StrOutputParser()

# chain=MODEL | parser

template="""
you are an assistant that provides answers to the questions based on
a given context.
Answer the question based on the context . If you can't answer then reply
with 'I don't know'.
Be concise.

Context: {context}
Question: {question}
"""

prompt=PromptTemplate.from_template(template)
# chain=prompt | MODEL | parser

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | prompt
    | MODEL
    | parser
)

st.title("WELCOME")
st.header("Ask a Question:")
ques=st.text_input(label="Enter here")

ans=chain.invoke({'question': ques})
st.write(ans)

# st.sidebar.header("Llama 3.1")
# st.sidebar.title("Choose your pdf").file_uploader()