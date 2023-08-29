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

# @st.cache(allow_output_mutation=True)
# def process_pdf(file):
    
#     # Create a virtual path for the file
#     virtual_directory = "/virtual_upload_directory"
#     unique_filename = f"{uuid.uuid4()}_{file.name}"
#     file_path = os.path.join(virtual_directory, unique_filename)
    
#     pages = read_pdf(uploaded_file, file_path)

#     text_splitter = CharacterTextSplitter(        
#         separator="\n\n",
#         chunk_size=2000,
#         chunk_overlap=500,
#         length_function=len,
#     )
#     splits = text_splitter.split_documents(pages)

#     with open("vectorstore.pkl", "wb") as f:
#         pickle.dump(vectorstore, f)

# Load Chain
chain = load_chain()

# Set Streamlit Config
st.set_page_config(page_title="ChatGPT for BERA", page_icon=":robot:")
st.header("ChatGPT for BERA")

# PDF Upload and Read
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Check if a file has been uploaded
if uploaded_file is not None:
    # Load vectorstore from pickle file if it exists
    if os.path.exists("vectorstore.pkl"):
        with open("vectorstore.pkl", "rb") as f:
            vectorstore = pickle.load(f)
    else:
        # Create a virtual path for the file
        virtual_directory = "/virtual_upload_directory"
        unique_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
        file_path = os.path.join(virtual_directory, unique_filename)

        # Now use the read_pdf function
        pages = read_pdf(uploaded_file, file_path)
        
        # Split PDF into chunks
        text_splitter = CharacterTextSplitter(        
            separator="\n\n",
            chunk_size=2000,
            chunk_overlap=500,
            length_function=len,
        )
        
        splits = text_splitter.split_documents(pages)

        chunk_texts = [chunk.page_content for chunk in splits]

        # Embedding (Openai methods)
        embeddings = OpenAIEmbeddings()
        
        # Store the chunks part in db (vector)
        vectorstore = FAISS.from_texts(
            texts=chunk_texts,  # Pass the extracted text content
            embedding=embeddings
        )
        
        # Store vectorstore to pickle file
        with open("vectorstore.pkl", "wb") as f:
            pickle.dump(vectorstore, f)
            
    question = st.text_input("Enter your question:", "Who are the main 3 findings?")
    if question:
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
    
        # Run chain
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.8)
        
# Handle user input and conversation history
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []

def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text

user_input = get_text()

if user_input:
    # Run the conversation chain with user input
    output = chain.run(input=user_input)

    # Append user input and generated output to session state
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

# Display conversation history
if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

# Handle the question input for the Question Answering part
question = st.text_input("Enter your question:", "Who are the main 3 findings?", key="question_input")
if question:
    result = qa_chain({"query": question})
    st.write(f"Answer: {result['result']}")

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
