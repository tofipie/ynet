import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
#from langchain.embeddings.openai import OpenAIEmbeddings
import pickle
from langchain.chains import ConversationalRetrievalChain
#from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from botocore.config import Config


import boto3
from langchain.llms.bedrock import Bedrock
from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from datetime import datetime

import numpy as np
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader

import json

# Load environment variables

# from retriver import load_local_vector_store



#boto3_bedrock = session.client("bedrock", config=retry_config)
#boto3_bedrock_runtime = session.client("bedrock-runtime", config=retry_config)



# Function to read PDF content
def read_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

pdf_mapping = {
    'Health Insurance Benefits': 'tax benefits due to health insurance.pdf',
    'Tax Regime':'new-regime-vs-old-regime.pdf',
    'שימור ידע - העברה מחשבון לחשבון' : 'שימור ידע - העברה מחשבון לחשבון.pdf',
    'שימור ידע - מחיקת חובות':'שימור ידע - מחיקת חובות.pdf',
    'שאילתות שומה ונכסים' :'שאילתות שומה ונכסים.pdf'
  #  'Reinforcement Learning': 'pdfs/SuttonBartoIPRLBook2ndEd.pdf',
   # 'GPT-4 All Training': 'pdfs/2023_GPT4All_Technical_Report.pdf',
}




# Main Streamlit app
def main():
    st.title("Chat PDF Using AWS Bedrock and Anthropic Claude")
    with st.sidebar:
        st.title('💬 PDF Chat App')
        st.markdown('''
        ## About
        בחר מסמך ולאחר מכן שאל שאלה
        ''')
        st.write('Made by Noa Cohen')
       
    custom_names = list(pdf_mapping.keys())

    selected_custom_name = st.sidebar.selectbox('בחר מסמך', ['', *custom_names])

    selected_actual_name = pdf_mapping.get(selected_custom_name)

    if selected_actual_name:
        pdf_folder = "pdfs"
        file_path = os.path.join(pdf_folder, selected_actual_name)

        try:
            text = read_pdf(file_path)
            st.info("הקלד שאלה")
        except FileNotFoundError:
            st.error(f"File not found: {file_path}")
            return
        except Exception as e:
            st.error(f"Error occurred while reading the PDF: {e}")
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )

        # Process the PDF text and create the documents list
        documents = text_splitter.split_text(text=text)

        # Vectorize the documents and create vectorstore
        #embeddings = OpenAIEmbeddings()

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},  )

        vectorstore = FAISS.from_texts(documents, embedding=embeddings)

        st.session_state.processed_data = {
            "document_chunks": documents,
            "vectorstore": vectorstore,
        }

        # Save vectorstore using pickle
       # pickle_folder = "Pickle"
       # if not os.path.exists(pickle_folder):
        #    os.mkdir(pickle_folder)

        #pickle_file_path = os.path.join(pickle_folder, f"{selected_custom_name}.pkl")

        #if not os.path.exists(pickle_file_path):
         #   with open(pickle_file_path, "wb") as f:
             #   pickle.dump(vectorstore, f)

        # Load the Langchain chatbot
       # llm = ChatOpenAI(temperature=0, max_tokens=1000, model_name="gpt-3.5-turbo")

        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl",
                    model_kwargs={"temperature":0.1, "max_length":512})
        
        qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

        # Initialize Streamlit chat UI
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                    st.markdown(message["content"])
             #   st.markdown('<div style="text-align: right;">message["content"]</div>',unsafe_allow_html=True)

               # st.markdown('<div style="text-align: justify;">Hello World!</div>', unsafe_allow_html=True)

        if prompt := st.chat_input("שאל שאלה על מסמך "f'{selected_custom_name}'):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                 st.markdown(prompt)
              #   st.markdown('<div style="text-align: right;">prompt</div>',unsafe_allow_html=True)
                    

            result = qa({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})
            print(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = result["answer"]
                message_placeholder.markdown(full_response + "|")
            message_placeholder.markdown(full_response)
            #message_placeholder.markdown(st.markdown('<div style="text-align: right;">full_response</div>',unsafe_allow_html=True))
            print(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
