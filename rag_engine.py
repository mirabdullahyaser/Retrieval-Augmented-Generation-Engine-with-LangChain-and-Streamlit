import os, tempfile
import pinecone
from pathlib import Path
from dotenv import load_dotenv
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain import OpenAI
from langchain_community.llms.openai import OpenAIChat
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma, Pinecone
# from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
# from langchain_community.chat_message_historie import StreamlitChatMessageHistory

import streamlit as st

load_dotenv()

TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

st.set_page_config(page_title="RAG")
st.title("Retrieval Augmented Generation Engine")


def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

def embeddings_on_local_vectordb(texts):
    vectordb = Chroma.from_documents(texts, embedding=OpenAIEmbeddings(),
                                     persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix())
    vectordb.persist()
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    return retriever

def embeddings_on_pinecone(texts):
    pinecone.init(api_key=st.session_state.pinecone_api_key, environment=st.session_state.pinecone_env)
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
    vectordb = Pinecone.from_documents(texts, embeddings, index_name=st.session_state.pinecone_index)
    retriever = vectordb.as_retriever()
    return retriever

def embedding_on_pinecone_new(texts):
    # from langchain_community.embeddings.openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
    from langchain_pinecone import Pinecone

    index_name = "quickstart"
    docsearch = Pinecone.from_documents(texts , embeddings, index_name=index_name)
    return docsearch.as_retriever()

def query_llm(retriever, query):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=OpenAIChat(openai_api_key=st.session_state.openai_api_key),
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
    result = result['answer']
    st.session_state.messages.append((query, result))
    return result

def query_llm_new(retriever, query):
    from langchain_openai import ChatOpenAI
    from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain

    with_source = True

    llm = ChatOpenAI(
        openai_api_key=st.session_state.openai_api_key,
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    result = qa_with_sources(query) if with_source else qa(query)
    st.session_state.messages.append((query, result))
    return result

def input_fields():
    #
    with st.sidebar:
        #
        if "openai_api_key" in st.secrets:
            st.session_state.openai_api_key = st.secrets.openai_api_key
        else:
            st.session_state.openai_api_key = st.text_input("OpenAI API key", type="password")
        #
        if "pinecone_api_key" in st.secrets:
            st.session_state.pinecone_api_key = st.secrets.pinecone_api_key
        else:
            st.session_state.pinecone_api_key = st.text_input("Pinecone API key", type="password")
        #
        if "pinecone_env" in st.secrets:
            st.session_state.pinecone_env = st.secrets.pinecone_env
        else:
            st.session_state.pinecone_env = st.text_input("Pinecone environment")
        #
        if "pinecone_index" in st.secrets:
            st.session_state.pinecone_index = st.secrets.pinecone_index
        else:
            st.session_state.pinecone_index = st.text_input("Pinecone index name")
    #
    st.session_state.pinecone_db = st.toggle('Use Pinecone Vector DB')
    #
    st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="pdf", accept_multiple_files=True)
    #


def process_documents():
    if not st.session_state.openai_api_key or not st.session_state.pinecone_api_key or not st.session_state.pinecone_env or not st.session_state.pinecone_index or not st.session_state.source_docs:
        st.warning(f"Please upload the documents and provide the missing fields.")
    else:
        try:
            for source_doc in st.session_state.source_docs:
                #
                with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.pdf') as tmp_file:
                    tmp_file.write(source_doc.read())
                #
                documents = load_documents()
                #
                for _file in TMP_DIR.iterdir():
                    temp_file = TMP_DIR.joinpath(_file)
                    temp_file.unlink()
                #
                texts = split_documents(documents)
                #
                if not st.session_state.pinecone_db:
                    st.session_state.retriever = embeddings_on_local_vectordb(texts)
                else:
                    st.session_state.retriever = embedding_on_pinecone_new(texts)
        except Exception as e:
            st.error(f"An error occurred: {e}")

def boot():
    #
    input_fields()
    #
    st.button("Submit Documents", on_click=process_documents)
    #
    if "messages" not in st.session_state:
        st.session_state.messages = []
    #
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])
    #
    if query := st.chat_input():
        st.chat_message("human").write(query)
        response = query_llm_new(st.session_state.retriever, query)
        st.chat_message("ai").write(response)

if __name__ == '__main__':
    #
    boot()
