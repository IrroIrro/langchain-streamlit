"""Python file to serve as the frontend"""
import PyPDF2

import streamlit as st
from streamlit_chat import message

from langchain.chains import ConversationChain
from langchain.llms import OpenAI

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def load_chain():
    llm = OpenAI(temperature=0.2)
    chain = ConversationChain(llm=llm)
    return chain
    
def process_paragraphs(paragraphs):
    results = []
    for paragraph in paragraphs:
        # Replace this with actual processing, e.g., model inference with LangChain.
        results.append(paragraph[::-1])
    return results

chain = load_chain()
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

# Upload and Read PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file:
    pdf_text = read_pdf(uploaded_file)
    st.text_area("Content of the PDF:", pdf_text, height=300)

    paragraphs = [para for para in pdf_text.split(". \n") if para]
    processed_results = process_paragraphs(paragraphs)
    for result in processed_results:
        st.write(result)

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
