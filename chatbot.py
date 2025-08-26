import streamlit as st
import os
import chromadb
from chromadb.utils import embedding_functions
import tempfile
import pdfplumber
import uuid
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_community.chat_models import ChatOpenAI

import torch
torch.classes.__path__ = []

def load_text_from_pdf(file_bytes) -> str:
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        tmp_path = tmp.name
    try:
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return text

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - chunk_overlap
    return chunks

st.set_page_config(page_title="Chatbot", layout="wide")
st.title("Chatbot with In Memory RAG")

if 'chroma_client' not in st.session_state:
    try:
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        st.error("Failed to initialize embedding function. Make sure sentence-transformers is installed.")
        raise

    client = chromadb.Client()

    collection = client.create_collection(name="documents", embedding_function=embedding_fn)

    st.session_state['chroma_client'] = client
    st.session_state['collection'] = collection
    st.session_state['embedding_fn'] = embedding_fn
    st.session_state['embedding_model_name'] = "all-MiniLM-L6-v2"
    st.session_state['docs_count'] = 0

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

with st.sidebar:
    st.header("Data & Settings")
    uploaded_files = st.file_uploader("Upload text / pdf files", accept_multiple_files=True, type=['txt','pdf','md'])
    pasted = st.text_area("Or paste text here")
    chunk_size = st.number_input("Chunk size (words)", min_value=100, max_value=2000, value=500, step=50)
    chunk_overlap = st.number_input("Chunk overlap (words)", min_value=0, max_value=500, value=50, step=10)
    top_k = st.number_input("Top-k retrieval", min_value=1, max_value=10, value=4)
    
    api_key = ''
    openai_api_key = st.text_input("API Key", type="password")
    
    if openai_api_key:
        api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = api_key
    
    if st.button("Index uploaded/pasted data"):
        docs_to_add = []
        metadatas = []
        ids = []
        # handle uploaded files
        for f in uploaded_files or []:
            fname = f.name
            content = None
            if fname.lower().endswith('.pdf'):
                content = load_text_from_pdf(f.getvalue())
            else:
                try:
                    content = f.getvalue().decode('utf-8')
                except Exception:
                    content = str(f.getvalue())
            if content:
                chunks = chunk_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                for i, c in enumerate(chunks):
                    docs_to_add.append(c)
                    metadatas.append({"source": fname, "chunk": i})
                    ids.append(str(uuid.uuid4()))

        # handle pasted text
        if pasted and pasted.strip():
            chunks = chunk_text(pasted, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for i, c in enumerate(chunks):
                docs_to_add.append(c)
                metadatas.append({"source": "pasted_text", "chunk": i})
                ids.append(str(uuid.uuid4()))

        if len(docs_to_add) == 0:
            st.warning("No documents or text to index. Upload files or paste text first.")
        else:
            # add to collection. When collection has embedding_function set, chroma will create embeddings.
            try:
                st.session_state['collection'].add(documents=docs_to_add, metadatas=metadatas, ids=ids)
                st.session_state['docs_count'] += len(docs_to_add)
                st.success(f"Indexed {len(docs_to_add)} chunks — total indexed: {st.session_state['docs_count']}")
            except Exception as e:
                st.error(f"Failed to add documents to ChromaDB: {e}")

    if st.button("Clear indexed data"):
        try:
            # delete collection and recreate
            st.session_state['chroma_client'].delete_collection(name="documents")
        except Exception:
            pass
        st.session_state['collection'] = st.session_state['chroma_client'].create_collection(name="documents", embedding_function=st.session_state['embedding_fn'])
        st.session_state['docs_count'] = 0
        st.success("Cleared indexed data.")
    
col1, col2 = st.columns([3,1])
with col1:   

    user_input = st.text_area("Ask a question", height=300)
    
    if st.button("Send") and user_input:
        st.session_state['chat_history'].append(("user", user_input))
        
        try:
            results = st.session_state['collection'].query(query_texts=[user_input], n_results=top_k, include=['documents','metadatas','distances'])
            hits = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results.get('distances', [None])[0]
        except Exception as e:
            st.error(f"Retrieval failed: {e}")
            hits = []
            metadatas = []
            distances = []
            
        # assemble context
        context_texts = []
        sources = []
        for doc, meta, dist in zip(hits, metadatas, distances if distances is not None else [None]*len(hits)):
            context_texts.append(f"Source: {meta.get('source')} | chunk: {meta.get('chunk')}\n{doc}")
            sources.append(meta.get('source'))

        combined_context = "\n---\n".join(context_texts)

        if openai_api_key:
            try:
                llm = ChatOpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        openai_api_key=os.getenv("OPENAI_API_KEY"),
                        model="openai/gpt-4.1-nano",
                        temperature=0)
                
                prompt = (
                    "You are a helpful assistant. Use the context below to answer the user's question. If the context doesn't contain the answer, say so.\n\n"
                    f"CONTEXT:\n{combined_context}\n\nQUESTION:\n{user_input}\n\nAnswer concisely and cite the sources in square brackets like [source_name]."
                )
                
                response = llm.invoke(prompt)
                
                print(response)
                answer = response
                st.session_state['chat_history'].append(("assistant", answer))
                
            except Exception as e:
                st.error(f"OpenAI request failed: {e}")
                answer = None

    for role, text in st.session_state['chat_history']:
        if role == 'user':
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Bot:** {text}")