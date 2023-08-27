import os, tempfile
import pinecone
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

import streamlit as st

LOCAL_VECTORDB = False

# openai.api_key = os.getenv('OPENAI_API_KEY')
# PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
# PINECONE_API_ENV = 'gcp-starter'
# PINECONE_INDEX = 'research-assistant'

RESEARCH_PAPERS_DIR = Path(__file__).resolve().parent.parent.joinpath('data', 'papers')

# QUERY = 'suggest a simple method to analyze crash simulations with machine learning'


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

def embeddings_on_pinecone(texts, pinecone_api_key, pinecone_env, pinecone_index):
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
    vectordb = Pinecone.from_documents(texts, embeddings, index_name=pinecone_index)
    retriever = vectordb.as_retriever()
    return retriever

def query_llm(retriever, openai_api_key, query):
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=openai_api_key),
        retriever=retriever,
        return_source_documents=True,
        
    )
    result = qa_chain({'query': query})
    return result['result']

def streamlit_layout():
    st.subheader('AI Powered Research Assistant')
    #
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API key", type="password")
        pinecone_api_key = st.text_input("Pinecone API key", type="password")
        pinecone_env = st.text_input("Pinecone environment")
        pinecone_index = st.text_input("Pinecone index name")
    #
    source_docs = st.file_uploader("Upload Documents", type="pdf", label_visibility="collapsed", accept_multiple_files=True)
    query = st.text_input("Enter your question")
    return openai_api_key, pinecone_api_key, pinecone_env, pinecone_index, source_docs, query
    #
def boot():
    #
    openai_api_key, pinecone_api_key, pinecone_env, pinecone_index, source_docs, query = streamlit_layout()
    #
    if st.button("Submit"):
        if not openai_api_key or not pinecone_api_key or not pinecone_env or not pinecone_index or not source_docs or not query:
            st.warning(f"Please upload the documents and provide the missing fields.")
        else:
            try:
                for source_doc in source_docs:
                    #
                    with tempfile.NamedTemporaryFile(delete=False, dir=RESEARCH_PAPERS_DIR.as_posix()) as tmp_file:
                        tmp_file.write(source_doc.read())
                    #
                    documents = load_documents()
                    #
                    for _file in RESEARCH_PAPERS_DIR.iterdir():
                        temp_file = RESEARCH_PAPERS_DIR.joinpath(_file)
                        temp_file.unlink()
                    #
                    texts = split_documents(documents)
                    #
                    if LOCAL_VECTORDB:
                        retriever = embeddings_on_local_vectordb(texts)
                    else:
                        retriever = embeddings_on_pinecone(texts, pinecone_api_key, pinecone_env, pinecone_index)
                    #    
                    response = query_llm(retriever, openai_api_key, query)
                    st.success(response)
                    #
            except Exception as e:
                st.error(f"An error occurred: {e}")


if __name__ == '__main__':
    #
    boot()