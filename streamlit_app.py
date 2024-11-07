from llama_index.core import (
    VectorStoreIndex, 
    StorageContext, 
    Settings, 
    Document,
    ServiceContext
)
from llama_index.core.storage import StorageContext
from llama_index.core.indices.loading import load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
import openai
import os
import streamlit as st
import shutil
import time
from pathlib import Path
import pypdf

from dotenv import load_dotenv
load_dotenv()

# Configure settings
Settings.chunk_size = 512
Settings.chunk_overlap = 50

st.set_page_config(
    page_title="Chat with Custom Data",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
)

# Check for API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
    st.stop()

openai.api_key = api_key
st.title("Chat with Local Data")
st.info("Custom data loaded from local folder.")

# Define the index directory path
INDEX_DIRECTORY = "saved_index"

def save_index(index, directory=INDEX_DIRECTORY):
    """Save the index to the specified directory"""
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        index.storage_context.persist(persist_dir=directory)
        st.success("Index saved successfully!")
    except Exception as e:
        st.error(f"Error saving index: {str(e)}")

def clear_index():
    """Clear the saved index"""
    try:
        if os.path.exists(INDEX_DIRECTORY):
            shutil.rmtree(INDEX_DIRECTORY)
            st.success("Index cleared successfully!")
    except Exception as e:
        st.error(f"Error clearing index: {str(e)}")

def load_single_pdf(filepath):
    """Load a single PDF file with custom PDF loading logic"""
    try:
        st.write(f"Loading {Path(filepath).name}...")
        
        # Read PDF file
        with open(filepath, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            
            # Extract text from each page
            text_content = []
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content.append(page.extract_text())
            
            # Join all pages with proper spacing
            full_text = "\n\n".join(text_content)
            
            # Create a Document object
            doc = Document(text=full_text, metadata={"source": filepath})
            return [doc]
            
    except Exception as e:
        st.error(f"Error loading {Path(filepath).name}: {str(e)}")
        return None

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question based on the local presentations!"
        }
    ]

@st.cache_resource(show_spinner=False)

def load_data():
    """Load data and create/load index"""
    try:
        if os.path.exists(INDEX_DIRECTORY) and os.path.isfile(os.path.join(INDEX_DIRECTORY, "docstore.json")):
            with st.spinner(text='Loading saved index...'):
                storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIRECTORY)
                index = load_index_from_storage(storage_context)
                st.success("Loaded existing index successfully!")
        else:
            with st.spinner(text='Creating new index from documents...'):
                local_folder_path = "./Data"
                
                if not os.path.exists(local_folder_path):
                    st.error(f"Data directory not found: {local_folder_path}")
                    return None

                docs = []
                pdf_files = [f for f in os.listdir(local_folder_path) if f.endswith('.pdf')]
                
                if not pdf_files:
                    st.warning("No PDF files found in the data directory.")
                    return None

                # Load documents with progress bar
                progress_bar = st.progress(0)
                for idx, filename in enumerate(pdf_files):
                    filepath = os.path.join(local_folder_path, filename)
                    
                    # Show file size
                    file_size = os.path.getsize(filepath) / (1024 * 1024)  # Convert to MB
                    st.write(f"Processing {filename} ({file_size:.2f} MB)")
                    
                    pdf_docs = load_single_pdf(filepath)
                    
                    if pdf_docs:
                        docs.extend(pdf_docs)
                    
                    # Update progress
                    progress = (idx + 1) / len(pdf_files)
                    progress_bar.progress(progress)
                
                if not docs:
                    st.error("No documents were successfully loaded.")
                    return None

                st.info(f"Creating index from {len(docs)} documents...")
                
                # Create the index with custom settings
                service_context = ServiceContext.from_defaults(
                    chunk_size=512,
                    chunk_overlap=50,
                    node_parser=SentenceSplitter(chunk_size=512)
                )
                
                index = VectorStoreIndex.from_documents(
                    docs,
                    service_context=service_context,
                    show_progress=True
                )
                
                save_index(index)
                st.success("Created and saved new index successfully!")

        return index

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        # Remove exc_info parameter and just log the full error message
        st.error(f"Full error: {str(e)}")
        return None

# Add debugging information in sidebar
st.sidebar.title("Debug Information")
st.sidebar.write(f"OpenAI API Key set: {'Yes' if api_key else 'No'}")
st.sidebar.write(f"Index directory exists: {os.path.exists(INDEX_DIRECTORY)}")
st.sidebar.write(f"Data directory exists: {os.path.exists('./Data')}")

# Add a button to clear the index and reload data
if st.sidebar.button("Reload Data (Clear Index)"):
    clear_index()
    st.cache_resource.clear()
    st.experimental_rerun()

index = load_data()

if index is None:
    st.error("Failed to load or create the index. Please check your data and try again.")
else:
    if 'chat_engine' not in st.session_state:
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode="condense_question",
            verbose=True,
        )

    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({'role': 'user', 'content': prompt})

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.write(message['content'])

    if st.session_state.messages[-1]['role'] != 'assistant':
        with st.chat_message('assistant'):
            with st.spinner('Thinking...'):
                response = st.session_state.chat_engine.chat(prompt)
                st.write(response.response)
                message = {'role': 'assistant', 'content': response.response}
                st.session_state.messages.append(message)