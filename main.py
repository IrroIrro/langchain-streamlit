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

# Load Chain
chain = load_chain()

# Set Streamlit Config
st.set_page_config(page_title="ChatGPT for BERA", page_icon=":robot:")
st.header("ChatGPT for BERA")

# PDF Upload and Read
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Check if a file has been uploaded
if uploaded_file is not None:
    # Create a virtual path for the file
    virtual_directory = "/virtual_upload_directory"
    unique_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
    file_path = os.path.join(virtual_directory, unique_filename)

    try:
        with st.spinner('Reading and processing PDF...'):
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

            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=OpenAIEmbeddings(),
                persist_directory=persist_directory
            )

question = st.text_input("Enter your question:", "Who are the main 3 findings?")
if question:
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use five sentences maximum and provide the most accurate and precise answer.
    {context}
    """        
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.8)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True
    )
    result = qa_chain({"query": question})

    st.write(f"Answer: {result['result']}")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text

user_input = get_text()

if user_input:
    output = chain.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
