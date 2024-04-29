import streamlit as st
from llama_index.legacy import ServiceContext
from llama_index.legacy.llms import OpenAI
from llama_index.legacy.retrievers import BM25Retriever
from llama_index.legacy.retrievers import VectorIndexRetriever
from llama_index.legacy.retrievers import BaseRetriever
from llama_index.legacy.chat_engine import CondensePlusContextChatEngine
from llama_index.legacy.query_engine import RetrieverQueryEngine
from llama_index.legacy.postprocessor import LongContextReorder
from llama_index.legacy.embeddings import OpenAIEmbedding
from llama_index.legacy import (StorageContext,load_index_from_storage)
import openai
import os
from index import indexgenerator

st.set_page_config(page_title="Chat with a book, powered by AIXplorers", page_icon="✅", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = os.environ['SECRET_TOKEN']
st.title("Chat with The Four Realms of Existence!! 💬")

with st.sidebar:
    st.title('🤗💬The Four Realms of Existence @ Chat Bot')
    st.success('Access to this Gen-AI Powered Chatbot is provided by  [Anupam](https://www.linkedin.com/in/anupamisb/)!!', icon='✅')
    hf_email = 'anupam_purwar2019@pgp.isb.edu'
    hf_pass = 'PASS'
    
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question from the book!!"}
    ]

indexPath=r"large_pdf_index"
m=["gpt-4-1106-preview","gpt-4-0125-preview"]
embed_model = OpenAIEmbedding(model="text-embedding-3-large")
#documentsPath=r"FinTech for Billions - Bhagwan Chowdhry & Syed Anas Ahmed.pdf"
storage_context = StorageContext.from_defaults(persist_dir=indexPath)
index = load_index_from_storage(storage_context,service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4-1106-preview", temperature=0),embed_model=embed_model))
#index=indexgenerator(indexPath,documentsPath)
vector_retriever = VectorIndexRetriever(index=index,similarity_top_k=5)
bm25_retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=2)
postprocessor = LongContextReorder()
class HybridRetriever(BaseRetriever):
    def __init__(self,vector_retriever, bm25_retriever):
        self.vector_retriever_2000 = vector_retriever
        self.bm25_retriever_2000 = bm25_retriever
        super().__init__()

    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever_2000.retrieve(query, **kwargs)
        vector_nodes = self.vector_retriever_2000.retrieve(query, **kwargs)
        all_nodes = bm25_nodes + vector_nodes
        return all_nodes
hybrid_retriever=HybridRetriever(vector_retriever,bm25_retriever)
llm = OpenAI(model="gpt-4-1106-preview") 
#llm = OpenAI(model=m[1])
#service_context = ServiceContext.from_defaults(llm=llm)
embed_model = OpenAIEmbedding(model="text-embedding-3-large")
service_context = ServiceContext.from_defaults(llm=OpenAI(model=m[1], temperature=0),embed_model=embed_model)
query_engine=RetrieverQueryEngine.from_args(retriever=hybrid_retriever,service_context=service_context,verbose=True)
if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = CondensePlusContextChatEngine.from_defaults(query_engine)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
