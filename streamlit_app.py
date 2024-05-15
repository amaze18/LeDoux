import streamlit as st
from llama_index.legacy import ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.legacy.retrievers import BM25Retriever
from llama_index.legacy.retrievers import VectorIndexRetriever
from llama_index.legacy.retrievers import BaseRetriever
from llama_index.legacy.chat_engine import CondensePlusContextChatEngine
from llama_index.legacy.query_engine import RetrieverQueryEngine
from llama_index.legacy.postprocessor import LongContextReorder
from llama_index.legacy.embeddings import OpenAIEmbedding
from llama_index.legacy.schema import MetadataMode
from llama_index.legacy import (StorageContext,load_index_from_storage)
from llama_index.legacy.core.llms.types import ChatMessage, MessageRole
from llama_index.legacy.schema import QueryBundle
import openai
import os
from index import indexgenerator

st.set_page_config(page_title="Chat with a book, powered by AIXplorers", page_icon="✅", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = os.environ['SECRET_TOKEN']
st.title("Chat with The Four Realms of Existence!! 💬")



DEFAULT_CONTEXT_PROMPT_TEMPLATE_1 = """
 You're an AI assistant to help students learn their course material via convertsations.
 The following is a friendly conversation between a user and an AI assistant for answering questions related to query.
 The assistant is talkative and provides lots of specific details in form of bullet points or short paras from the context.
 Here is the relevant context:
 {context_str}
 Instruction: Based on the above context, provide a detailed answer IN THE USER'S LANGUAGE with logical formation of paragraphs for the user question below.
 """

DEFAULT_CONTEXT_PROMPT_TEMPLATE_2 = """
  The following is a friendly conversation between a user and an AI assistant.
  The assistant is talkative and provides lots of specific details from its context only.
  Here are the relevant documents for the context:
  {context_str}
  Instruction: Based on the above context, provide a detailed answer with logical formation of paragraphs for the user question below.
  Answer "don't know" if information is not present in context. Also, decline to answer questions that are not related to context."
  """

with st.sidebar:
    st.title('🤗💬The Four Realms of Existence @ Chat Bot')
    st.success('Access to this Gen-AI Powered Chatbot is provided by  [Anupam](https://www.linkedin.com/in/anupamisb/)!!', icon='✅')
    hf_email = 'anupam_purwar2019@pgp.isb.edu'
    hf_pass = 'PASS'
    
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question from the book!!"}
    ]

if "message_history" not in st.session_state.keys():
    st.session_state.message_history=[ChatMessage(role=MessageRole.ASSISTANT,content="Ask me a question from the book!!"),]

indexPath=r"LeDou"
m=["gpt-4-1106-preview","gpt-4-0125-preview","gpt-4o"]
embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
llm = OpenAI("gpt-4o") 
#documentsPath=r"FinTech for Billions - Bhagwan Chowdhry & Syed Anas Ahmed.pdf"
storage_context = StorageContext.from_defaults(persist_dir=indexPath)
index = load_index_from_storage(storage_context, service_context = ServiceContext.from_defaults(llm=llm))
#index=indexgenerator(indexPath,documentsPath)
# vector_retriever = VectorIndexRetriever(index=index,similarity_top_k=5)
# bm25_retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=2)
topk= 10
vector_retriever = VectorIndexRetriever(index=index,similarity_top_k=topk)
postprocessor = LongContextReorder()
bm25_flag = True
try:
    bm25_retriever = BM25Retriever.from_defaults(index=index,similarity_top_k=8)
except:
    source_nodes = index.docstore.docs.values()
    nodes = list(source_nodes)
    bm25_flag = False
class HybridRetriever(BaseRetriever):
    def __init__(self,vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)
        all_nodes = bm25_nodes + vector_nodes
        query = str(query)
        all_nodes = postprocessor.postprocess_nodes(nodes=all_nodes,query_bundle=QueryBundle(query_str=query.lower()))
        return all_nodes[0:topk]

if bm25_flag:
    hybrid_retriever=HybridRetriever(vector_retriever,bm25_retriever)
else:
    hybrid_retriever=vector_retriever

# hybrid_retriever=HybridRetriever(vector_retriever,bm25_retriever)

llm = OpenAI("gpt-4o") 
#llm = OpenAI(model=m[1])
#service_context = ServiceContext.from_defaults(llm=llm)
embed_model = OpenAIEmbedding(model="text-embedding-3-large")
service_context = ServiceContext.from_defaults(llm=OpenAI("gpt-4o"))
query_engine=RetrieverQueryEngine.from_args(retriever=hybrid_retriever,service_context=service_context,verbose=True)
if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = CondensePlusContextChatEngine.from_defaults(query_engine,context_prompt=DEFAULT_CONTEXT_PROMPT_TEMPLATE_1)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            all_nodes  = hybrid_retriever.retrieve(str(prompt))
            response = st.session_state.chat_engine.chat(str(prompt))
            #response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            context_str = "\n\n".join([n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in all_nodes])
            st.session_state.message_history.append(ChatMessage(role=MessageRole.ASSISTANT,content=str(response.response)),)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
