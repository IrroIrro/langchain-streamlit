"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
import os
import uuid
from collections import namedtuple
import PyPDF2
import pickle
from langchain.vectorstores import FAISS
import tiktoken
import numpy as np

# Extend the Document structure to include a metadata attribute
Document = namedtuple("Document", ["page_content", "metadata"])

def load_chain():
    global llm
    llm = OpenAI(temperature=0.2)
    chain = ConversationChain(llm=llm)
    return chain

def read_pdf(file, file_path):
    pdf_reader = PyPDF2.PdfReader(file)
    pages_content = []
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        metadata = {'source': file_path, 'page': page_num + 1}
        pages_content.append(Document(page_content=page.extract_text(), metadata=metadata))
    return pages_content

def process_and_create_vectorstore(uploaded_file):
    # Split PDF into chunks
    text_splitter = CharacterTextSplitter(        
        separator="\n\n",
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len,
    )
    
    pages = read_pdf(uploaded_file, file_path)  # Assuming file_path is defined earlier
    
    splits = text_splitter.split_documents(pages)

    chunk_texts = [chunk.page_content for chunk in splits]

    # Embedding (Openai methods)
    embeddings = OpenAIEmbeddings()
    
    # Store the chunks part in db (vector)
    vectorstore = FAISS.from_texts(
        texts=chunk_texts,  # Pass the extracted text content
        embedding=embeddings)    
    return vectorstore

# Initialize Streamlit UI
st.set_page_config(page_title="ChatGPT for BERA", page_icon=":robot:")
st.header("ChatGPT for BERA")

chain = load_chain()

# Initialize session state for chat history and uploaded files
if "generated_qa" not in st.session_state:
    st.session_state["generated_qa"] = []
if "past_qa" not in st.session_state:
    st.session_state["past_qa"] = []
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    st.session_state.uploaded_files.append({"file": uploaded_file, "title": None})
    uploaded_file_title = st.text_input("Enter a title for the uploaded PDF file:")
    
    # Button to process the uploaded PDF
    if st.button("Process and work with PDF"):
        if uploaded_file_title:
            virtual_directory = "/virtual_upload_directory"
            unique_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
            file_path = os.path.join(virtual_directory, unique_filename)
            vectorstore = process_and_create_vectorstore(uploaded_file)
            vectorstore_filename = f"vectorstore_{uploaded_file_title}.pkl"
            
            with open(vectorstore_filename, "wb") as f:
                pickle.dump(vectorstore, f)
            
vectorstore_files = [filename for filename in os.listdir() if filename.startswith("vectorstore_")]
vectorstore_titles = [filename[len("vectorstore_"):-len(".pkl")] for filename in vectorstore_files]

if vectorstore_titles:
    selected_title = st.selectbox("Select a stored PDF file:", vectorstore_titles)
    
    if st.button("Load Selected File"):
        selected_filename = f"vectorstore_{selected_title}.pkl"
        
        with open(selected_filename, "rb") as f:
            vectorstore = pickle.load(f)
        
        # Load the QA chain once the vectorstore is loaded
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever(), 
                                               chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}, 
                                               return_source_documents=True)
        question = st.text_input("Enter your question about the document:")
        
        if question:
            result = qa_chain({"query": question})
            st.session_state["generated_qa"].append(result['result'])
            st.session_state["past_qa"].append(question)
            
    # Display conversation history for QA
    if st.session_state["generated_qa"]:
        for i in range(len(st.session_state["generated_qa"])):
            message(st.session_state["generated_qa"][i], key=f"{i}_generated_qa")
            message(st.session_state["past_qa"][i], is_user=True, key=f"{i}_user_qa")
