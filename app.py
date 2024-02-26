from tqdm import tqdm
from bs4 import BeautifulSoup
import requests
import json
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings,HuggingFaceInstructEmbeddings
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings,HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import streamlit as st
from utils import reset_conversation

st.title("חיפוש בתוך מסמכים בעזרת בינה מלאכותית")
st.sidebar.title("App Description")
with st.sidebar:
    st.button('New Chat', on_click=reset_conversation)
  
    st.write('Made by Noa Cohen')

def get_articles_from_ynet():
    url = "https://www.ynet.co.il"
    #url = "https://www.ynetnews.com" #ynet english
    #url = "https://supreme.court.gov.il/pages/fullsearch.aspx"
    #url = "https://www.gov.il/he/departments/dynamiccollectors/spokmanship_court?skip=0"

    articles = []
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    sub_links = []
    for a in soup.find_all('a', href=True):
        if "article" in str(a['href']):
            sub_links.append(a['href'])

    sub_links = set(sub_links)
    # print(sub_links)

    print(f"Importing articles from: {url}. Found {len(sub_links)} sub links.")
    for link in tqdm(sub_links):
        page = requests.get(link)
        soup = BeautifulSoup(page.content, 'html.parser')
        data = soup.find('script', attrs={'type': 'application/ld+json'})
        data = str(data)
        data = data.split("{", 1)[1]
        d = data.strip("</script>")
        d = "{" + d
        try:
            metaData = json.loads(d)
        except Exception as e:
            # print("Error loading article meta-data to json ", e)
            continue

        try:
            authors = metaData['author']['name'].split(',')  # to list
            keywords = metaData['keywords'].split(',')  # to list
            date = metaData['datePublished']

            article = {
                'title': metaData['headline'],
                'date_published': date,
                'authors': authors,
                'link': link,
                'keywords': keywords,
                'summary': metaData['description'],
                'text': metaData['articleBody'],
                'link': link
            }
        except Exception as e:
            # print("Error loading article data ", e)
            continue

        if article['text']:
            articles.append(article)

    return articles

articles = get_articles_from_ynet()

docs = []
for article in articles:
    doc =  Document(
        page_content=article['text'],
        metadata={
            "title": article['title'],
            "link": article['link'],
            "authors": article['authors'],
            "date_published": article['date_published'],
            "summary": article['summary'],
            "keywords": article['keywords']
        }
    )
    docs.append(doc)


text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=100,
)
#docs = [d.page_content for d in docs]

splits = text_splitter.split_documents(docs)
texts = [d.page_content for d in splits]


embeddings = HuggingFaceEmbeddings(
        model_name="imvladikon/sentence-transformers-alephbert",
        model_kwargs={"device": "cpu"},
    )

texts = texts[:5]
# initialize the bm25 retriever and faiss retriever

faiss_vectorstore = FAISS.from_texts(texts,embeddings) #for documents type
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 5})

bm25_retriever = BM25Retriever.from_texts(texts)
bm25_retriever.k = 5

# initialize the ensemble retriever
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever],
                                       weights=[0.5, 0.5])




##############



prompt = st.chat_input("בצע שאילתת חיפוש...")


