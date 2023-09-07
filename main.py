import os
import tempfile
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import PyPDF2
from collections import namedtuple
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import io

# Handle parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(page_title="BERA: Chat with Documents", page_icon="🦜")
st.title("BERA: Chat with PDFs")

# Initialize the session state if not already done
if 'uploaded_pdfs' not in st.session_state:
    st.session_state.uploaded_pdfs = []
if 'langchain_messages' not in st.session_state:
    st.session_state.langchain_messages = []

st.sidebar.header('Document Source')
uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Reset the session state for selected files when new files are uploaded
if uploaded_files:
    st.session_state.uploaded_pdfs = [file.name for file in uploaded_files]

selected_files = st.sidebar.multiselect("Select from uploaded PDFs:", st.session_state.uploaded_pdfs)

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()
    
def read_pdfs(file):
    # Use PyPDF2 to extract text from the PDF
    with io.BytesIO(file.getvalue()) as open_pdf_file:
        pdf_reader = PyPDF2.PdfReader(open_pdf_file)
        number_of_pages = len(pdf_reader.pages)
        text_pages = [Page(pdf_reader.pages[i].extract_text(), metadata={"page_number": i + 1}) for i in range(number_of_pages)]
    return text_pages

def old_version_retriever(uploaded_files):
    # Initialize an empty list to store all pages from all uploaded files
    all_pages = []

    for uploaded_file in uploaded_files:
        # Convert uploaded file to in-memory binary stream
        if isinstance(uploaded_file, str):
            in_memory_file = io.BytesIO(uploaded_file.encode())  # Convert string to bytes and then to BytesIO object
        else:
            in_memory_file = io.BytesIO(uploaded_file.getvalue())
                
        # Extract pages from the current uploaded file
        pages = read_pdfs(in_memory_file)
        
        # Append the extracted pages to the all_pages list
        all_pages.extend(pages)
    
    # Split text from all_pages using your splitter
    text_splitter = CharacterTextSplitter(        
        separator="\n\n",
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len,
    )
    splits = text_splitter.split_documents(all_pages)
    
    # Rest of your function remains the same
    chunk_texts = [chunk.page_content for chunk in splits]
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(
        texts=chunk_texts,
        embedding=embeddings
    )    
    retriever = vectorstore.as_retriever()

    return retriever
    
class Page:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

from langchain.memory.chat_message_histories import StreamlitChatMessageHistory as LangChainStreamlitChatMessageHistory

class StreamlitChatMessageHistory(LangChainStreamlitChatMessageHistory):
    def __init__(self, key="langchain_messages"):
        super().__init__(key=key)
        
        # Ensure the key exists in the session state
        if self.key not in st.session_state:
            st.session_state[self.key] = []
        
        self._messages = st.session_state[self.key]

    def add_user_message(self, message):
        self._messages.append({"type": "user", "content": message})

    def add_ai_message(self, message):
        self._messages.append({"type": "assistant", "content": message})

    @property
    def messages(self):
        return self._messages

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)

class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")

if not uploaded_files and not selected_files:
    st.info("Please upload or select PDF documents to continue.")
    st.stop()
    
# Add a select button for the select box
if st.sidebar.button("Confirm Selection"):
    if not selected_files:
        st.warning("Please select at least one file.")
    else:
        files_to_process = [file for file in uploaded_files if file.name in selected_files]
        retriever = old_version_retriever(files_to_process)
else:
    files_to_process = []

files_to_process = uploaded_files if uploaded_files else selected_files

if not files_to_process:
    st.info("Please upload or select PDF documents to continue.")
    st.stop()

retriever = old_version_retriever(files_to_process)

# Initial message if no messages exist
msgs = StreamlitChatMessageHistory()
if not msgs.messages or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")
    
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Setup LLM and QA chain
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", openai_api_key = openai_api_key, temperature=0.5, streaming=True
)

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

qa_chain = RetrievalQA.from_chain_type(llm,
                           retriever=retriever,
                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
                           return_source_documents=True)  


# Displaying the chat history
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# Chat interaction
if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)
    
    # Retrieve the result and display
    result = qa_chain({"query": user_query})['result']
    st.chat_message("assistant").write(result)
