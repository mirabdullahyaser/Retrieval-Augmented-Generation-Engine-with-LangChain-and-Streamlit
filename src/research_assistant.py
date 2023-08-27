import os, tempfile
import pinecone
from pathlib import Path

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

import streamlit as st

LOCAL_VECTORDB = False
RESEARCH_PAPERS_DIR = Path(__file__).resolve().parent.parent.joinpath('data', 'papers')

st.set_page_config(page_title="AI-Powered-Research-Assistant", page_icon="📖")
st.title("AI Powered Research Assistant")


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
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=OpenAI(openai_api_key=openai_api_key),
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
    result = result['answer']
    st.session_state.messages.append((query, result))
    return result

def streamlit_layout():
    st.subheader('Upload Documents')
    #
    with st.sidebar:
        #
        if "openai_api_key" in st.secrets:
            openai_api_key = st.secrets.openai_api_key
        else:
            openai_api_key = st.text_input("OpenAI API key", type="password")
        #
        if "pinecone_api_key" in st.secrets:
            pinecone_api_key = st.secrets.pinecone_api_key
        else: 
            pinecone_api_key = st.text_input("Pinecone API key", type="password")
        #
        if "pinecone_env" in st.secrets:
            pinecone_env = st.secrets.pinecone_env
        else:
            pinecone_env = st.text_input("Pinecone environment")
        #
        if "pinecone_index" in st.secrets:
            pinecone_index = st.secrets.pinecone_index
        else:
            pinecone_index = st.text_input("Pinecone index name")
    #
    source_docs = st.file_uploader("Upload Documents", type="pdf", label_visibility="collapsed", accept_multiple_files=True)
    return openai_api_key, pinecone_api_key, pinecone_env, pinecone_index, source_docs
    #
def boot():
    #
    openai_api_key, pinecone_api_key, pinecone_env, pinecone_index, source_docs = streamlit_layout()
    #
    if "messages" not in st.session_state:
        st.session_state.messages = []    
    #
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    
    #
    # if st.button("Submit"):
    if not openai_api_key or not pinecone_api_key or not pinecone_env or not pinecone_index or not source_docs:
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
                if query := st.chat_input():
                    st.chat_message("human").write(query)
                    response = query_llm(retriever, openai_api_key, query)
                    st.chat_message("ai").write(response)
                #
        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == '__main__':
    #
    boot()