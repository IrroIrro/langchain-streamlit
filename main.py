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

def read_pdf(file, file_path):
    pdf_reader = PyPDF2.PdfReader(file)
    pages_content = []
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        metadata = {'source': file_path, 'page': page_num + 1}
        pages_content.append(Document(page_content=page.extract_text(), metadata=metadata))
    return pages_content

def load_chain():
    llm = OpenAI(temperature=0.2)
    chain = ConversationChain(llm=llm)
    return chain

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

# Load Chain
chain = load_chain()

# Load QA chain
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

template = """Based on the following excerpts from scientific papers, provide an answer to the question that follows.
        Structure your answer with a minimum of two paragraphs, each containing at least five sentences. Begin by presenting a general overview and then delve into specific details, such as numerical data or particular citations.
        If the answer is not apparent from the provided context, state explicitly that you don't have the information. 
        When referencing the content, provide a scientific citation. For instance: (Blom and Voesenek, 1996).
        If the source is unknown, indicate with "No Source".
        
        Context:
        {context}
        
        Question: 
        {question}
        
        Desired Answer:"""
        
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

# Set Streamlit Config
st.set_page_config(page_title="ChatGPT for BERA", page_icon=":robot:")
st.header("ChatGPT for BERA")

# Create a placeholder for the content
content_placeholder = st.empty()

# Display a default prompt
content_placeholder.text("How are you? Please choose or upload a PDF file.")

# Initialize variables
vectorstore_titles = []
uploaded_file = None
uploaded_file_title = None

# PDF Upload and Read
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Display user-defined title input if not already defined
if uploaded_file is not None and uploaded_file_title is None:
    uploaded_file_title = st.text_input("Enter a title for the uploaded PDF file:")
    
    if uploaded_file_title:
        vectorstore_titles.append(uploaded_file_title)  # Add the title to the list
    
    virtual_directory = "/virtual_upload_directory"
    unique_filename = f"{uuid.uuid4()}_{uploaded_file_title.name}"
    file_path = os.path.join(virtual_directory, unique_filename)

    vectorstore = process_and_create_vectorstore(uploaded_file)

    if uploaded_file_title:
        vectorstore_filename = f"vectorstore_{uploaded_file_title}.pkl"
    # else:
    #     vectorstore_filename = f"vectorstore_{uuid.uuid4()}.pkl"

    with open(vectorstore_filename, "wb") as f:
        pickle.dump(vectorstore, f)

# Display dropdown with user-friendly vectorstore titles
vectorstore_files = [filename for filename in os.listdir() if filename.startswith("vectorstore_")]
vectorstore_titles = [filename[len("vectorstore_"):-len(".pkl")] for filename in vectorstore_files]
selected_title = st.selectbox("Select a stored PDF file:", vectorstore_titles)

# Remove the selected title from the list to avoid duplication
if selected_title:
    vectorstore_titles.remove(selected_title)

    # Load the selected vectorstore based on the user-friendly title
    selected_filename = f"vectorstore_{selected_title}.pkl"
    with open(selected_filename, "rb") as f:
        vectorstore = pickle.load(f)

    # Display remove button
    if st.button("Remove this stored PDF file"):
        os.remove(selected_filename)

if uploaded_file is not None or selected_title is not None:                 
    # Create the QA chain after vectorstore is available
    qa_chain = RetrievalQA.from_chain_type(llm,
                       retriever=vectorstore.as_retriever(),
                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
                       return_source_documents=True)
    
    
    # Display conversation history for QA
    if "generated_qa" not in st.session_state:
        st.session_state["generated_qa"] = []
    if "past_qa" not in st.session_state:
        st.session_state["past_qa"] = []
    
    # Handle the question input for the Question Answering part
    question = st.text_input("Enter your question about the document:", key="question_input")
    if question:
        result = qa_chain({"query": question})
        st.write(f"Answer: {result['result']}")
    
    # Display conversation history for QA
    if st.session_state["generated_qa"]:
        for i in range(len(st.session_state["generated_qa"]) - 1, -1, -1):
            message(st.session_state["generated_qa"][i], key=f"{i}_generated_qa")
            message(st.session_state["past_qa"][i], is_user=True, key=f"{i}_user_qa")
