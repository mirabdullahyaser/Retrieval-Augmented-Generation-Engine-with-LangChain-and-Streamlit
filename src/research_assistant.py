import os
import openai
import pinecone
from pathlib import Path

import langchain
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

import streamlit as st

import nltk
nltk.download('punkt')

LOCAL_VECTORDB = False

openai.api_key = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = 'gcp-starter'
PINECONE_INDEX = 'research-assistant'

RESEARCH_PAPERS_DIR = Path(__file__).resolve().parent.parent.joinpath('data', 'papers')

QUERY = 'suggest a simple method to analyze crash simulations with machine learning'


def load_documents():
    loader = DirectoryLoader(RESEARCH_PAPERS_DIR, glob='**/*.pdf')
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

def embeddings_on_local_vectordb(texts):
    vectordb = Chroma.from_documents(texts, embedding=OpenAIEmbeddings(),
                                     persist_directory='./../data/vector_store')
    vectordb.persist()
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    return retriever

def embeddings_on_pinecone(texts):
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
    vectordb = Pinecone.from_documents(texts, embeddings, index_name=PINECONE_INDEX)
    retriever = vectordb.as_retriever()
    return retriever

def query_llm(retriever):
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        retriever=retriever,
        return_source_documents=True
    )
    result = qa_chain({'query': QUERY})
    print(result['result'])

def boot():
    #
    st.subheader('Research Assistant')
    #
    # documents = load_documents()
    # #
    # texts = split_documents(documents)
    # #
    # if LOCAL_VECTORDB:
    #     retriever = embeddings_on_local_vectordb(texts)
    # else:
    #     retriever = embeddings_on_pinecone(texts)
    # #
    # query_llm(retriever)


if __name__ == '__main__':
    #
    boot()