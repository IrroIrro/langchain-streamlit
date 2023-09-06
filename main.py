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

st.set_page_config(page_title="BERA: Chat with Documents", page_icon="🦜")
st.title("BERA: Chat with PDFs")

# Initialize session state for langchain_messages
if 'langchain_messages' not in st.session_state:
    st.session_state.langchain_messages = []

# Store the PDFs in a list for selection
available_pdfs = []  # Example list

# At the very beginning
if 'uploaded_pdfs' not in st.session_state:
    st.session_state.uploaded_pdfs = []
    
# Sidebar to upload or select files
st.sidebar.header('Document Source')
uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    # Use a set to ensure unique filenames
    st.session_state.uploaded_pdfs = list(set(st.session_state.uploaded_pdfs + [file.name for file in uploaded_files]))

available_pdfs = available_pdfs + st.session_state.uploaded_pdfs
selected_files = st.sidebar.multiselect("Or select from existing PDFs:", available_pdfs)

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()
    
if not uploaded_files and not selected_files:
    st.info("Please upload or select PDF documents to continue.")
    st.stop()

def configure_retriever(uploaded_files):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})

    return retriever

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

files_to_process = uploaded_files if uploaded_files else selected_files
retriever = configure_retriever(files_to_process)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Setup LLM and QA chain
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", openai_api_key = openai_api_key, temperature=0.5, streaming=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True
)

# template = """Based on the following excerpts from scientific papers, provide an answer to the question that follows.
#         Structure your answer with a minimum of two paragraphs, each containing at least five sentences. Begin by presenting a general overview and then delve into specific details, such as numerical data or particular citations.
#         If the answer is not apparent from the provided context, state explicitly that you don't have the information. 
#         When referencing the content, provide a scientific citation. For instance: (Blom and Voesenek, 1996).
#         If the source is unknown, indicate with "No Source".
        
#         Context:
#         {context}
        
#         Question: 
#         {question}
        
#         Desired Answer:"""
        
# QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

# qa_chain = RetrievalQA.from_chain_type(llm,
#                            retriever=retriever,
#                            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
#                            return_source_documents=True)  


if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)
   
if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)
    
    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
        # result = qa_chain({"query": user_query})['result']
