# Retrieval Augmented Generation Engine using LangChain, Streamlit, & Pinecone

![Demo](data/demo.gif)

## Overview

The Retrieval Augmented Engine (RAG) is a powerful tool for document retrieval, summarization, and interactive question-answering. This project utilizes LangChain, Streamlit, and Pinecone to provide a seamless web application for users to perform these tasks. With RAG, you can easily upload multiple PDF documents, generate vector embeddings for text within these documents, and perform conversational interactions with the documents. The chat history is also remembered for a more interactive experience.


## Features

- **Streamlit Web App**: The project is built using Streamlit, providing an intuitive and interactive web interface for users.
- **Input Fields**: Users can input essential credentials like OpenAI API key and Pinecone API key through dedicated input fields.
- **Document Uploader**: Users can upload multiple PDF files, which are then processed for further analysis.
- **Document Splitting**: The uploaded PDFs are split into smaller text chunks, ensuring compatibility with models with token limits.
- **Vector Embeddings**: The text chunks are converted into vector embeddings, making it easier to perform retrieval and question-answering tasks.
- **Flexible Vector Storage**: You can choose to store vector embeddings either in Pinecone or a local vector store, providing flexibility and control.
- **Interactive Conversations**: Users can engage in interactive conversations with the documents, asking questions and receiving answers. The chat history is preserved for reference.


## Prerequisites

Before running the project, make sure you have the following prerequisites:

- Python 3.7+
- LangChain
- Streamlit
- Pinecone
- An OpenAI API key
- PDF documents to upload

## Usage

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/mirabdullahyaser/Retrieval-Augmented-Generation-Engine-with-LangChain-and-Streamlit.git
   cd Retrieval-Augmented-Generation-Engine-with-LangChain-and-Streamlit
   ```

2. Install the required dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run src/rag_engine.py
   ```

4. Access the app by opening a web browser and navigating to the provided URL.

5. Input your OpenAI API key, Pinecone API key, Pinecone environment, and Pinecone index name in the respective fields. You can provide them either in the sidebar of the application or place them in the **secrets.toml** file in the [.streamlit directory](src/.streamlit)

6. Upload the PDF documents you want to analyze.

7. Click the "Submit Documents" button to process the documents and generate vector embeddings.

8. Engage in interactive conversations with the documents by typing your questions in the chat input box.

## Contributors

[Mir Abdullah Yaser](https://github.com/mirabdullahyaser)

## Contact

If you have any questions, suggestions, or would like to discuss this project further, feel free to get in touch with me:

- [Email](mailto:mirabdullahyaser@gmail.com)
- [LinkedIn](https://www.linkedin.com/in/mir-abdullah-yaser/)

I'm open to collaboration and would be happy to connect!