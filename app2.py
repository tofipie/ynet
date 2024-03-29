

from urllib.request import urlretrieve
import os
import streamlit as st
from utils import reset_conversation
import re
import time
from langchain.retrievers import BM25Retriever, EnsembleRetriever

st.title("חיפוש בתוך מסמכים בעזרת בינה מלאכותית")
st.sidebar.title("App Description")
with st.sidebar:
    st.button('New Chat', on_click=reset_conversation)
  
    st.write('Made by Noa Cohen')
    
os.makedirs("data", exist_ok=True)
files = [
    "https://www.irs.gov/pub/irs-pdf/p1544.pdf",
    "https://www.irs.gov/pub/irs-pdf/p15.pdf",
    "https://www.irs.gov/pub/irs-pdf/p1212.pdf",
]
for url in files:
    file_path = os.path.join("data", url.rpartition("/")[2])
    urlretrieve(url, file_path)

###
#DATA_PATH = "data/"
#loader = PyPDFDirectoryLoader("./pdfs/")

###

import numpy as np
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader

loader = PyPDFDirectoryLoader("./data/")

documents = loader.load()
# - in our testing Character split works better with this PDF data set
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1000,
    chunk_overlap  = 100,
)
docs = text_splitter.split_documents(documents)
docs = [d.page_content for d in docs]


from langchain.embeddings import HuggingFaceEmbeddings

bedrock_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
       model_kwargs={"device": "cpu"},
  )

from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator 
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

#vectorstore_faiss = FAISS.from_documents(   docs,    bedrock_embeddings,)

#wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)
###ENSEMBLE RETRIEVER


faiss_vectorstore = FAISS.from_texts(docs,bedrock_embeddings) #for text type
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 5})

bm25_retriever = BM25Retriever.from_texts(docs)
bm25_retriever.k = 5

# initialize the ensemble retriever
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever],
                                       weights=[0.5, 0.5])

####

# Create a function to generate a resposne from the model
def generate_response(input_text):        
   #This will initiate the LLM and run a similarity search across the input text on your documents
   # docs = faiss_retriever.similarity_search(input_text)
    docs = ensemble_retriever.get_relevant_documents(input_text)
    

   # Write the input text from the user onto the chat window
    with st.chat_message("user"):
        st.write(input_text)
        st.session_state.messages.append({"role": "user", "content": input_text})

    # Take the output message and display in the chat box
    with st.chat_message("assistant"):
       # st.toast("Running...", icon="⏳")

        response = docs[0].page_content#chain.run(input_documents = docs, question = input_text)
        response = response.replace('\n',' ')
        message_placeholder = st.empty()
        full_response = ""

        # Simulate stream of response with milliseconds delay. THis is not true streaming functionality. We use re.split functionality to ensure that line breaks are preserved in the output.
       # for chunk in re.split(r'(\s+)', response):
        #    full_response += chunk + " "
        #    time.sleep(0.05)
            # Add a blinking cursor to simulate typing
         #   message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(response) #full_response

        st.session_state.messages.append({"role": "assistant", "content": response})
#~~~~~~~~~~~~~~~~~~~~~~~
#if st.session_state['messages']:
 #   for i in range(len(st.session_state['messages'])):
  #      message(st.session_state['messages'][i], is_user=True, key=str(i) + '_user')
#~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create an input box to take the user''s input question
prompt = st.chat_input("שאל שאלה...")

if prompt:
    generate_response(prompt)
