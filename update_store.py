# --- update_store.py (Upgraded for Multi-Index Support) ---

import os
import shutil
from dotenv import load_dotenv
import time

# --- LangChain Imports ---
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

# --- NEW TIERED CONFIGURATION ---
# We define our knowledge sources and where to save their indexes.
# You can add more sources here in the future!
KNOWLEDGE_SOURCES = {
    "essays": {
        "source_path": "essays",
        "index_path": "faiss_index_essays" # The "Truth" brain
    },
    "dossiers": {
        "source_path": "dossiers",
        "index_path": "faiss_index_dossiers" # The "Perspectives" brain
    }
}
PROCESSED_LOG_PATH = "processed_files.log" # A single log for all sources
EMBEDDING_MODEL = "models/text-embedding-004"

def get_processed_files():
    """Reads the log file and returns a set of processed file paths."""
    if not os.path.exists(PROCESSED_LOG_PATH):
        return set()
    with open(PROCESSED_LOG_PATH, 'r', encoding='utf-8') as f:
        # We use os.path.normpath to handle different OS path styles (e.g., / vs \)
        return set(os.path.normpath(line.strip()) for line in f)

def update_processed_log(file_paths):
    """Appends a list of newly processed files to the log."""
    with open(PROCESSED_LOG_PATH, 'a', encoding='utf-8') as f:
        for path in file_paths:
            f.write(os.path.normpath(path) + '\n')

def process_single_source(source_name, source_info, processed_files):
    """Builds or updates a vector store for ONE knowledge source."""
    source_path = source_info["source_path"]
    index_path = source_info["index_path"]
    
    print(f"\n{'='*20} PROCESSING SOURCE: {source_name.upper()} {'='*20}")

    # 1. Find new documents for THIS source
    all_source_files = []
    if os.path.isdir(source_path):
        for filename in os.listdir(source_path):
            if filename.endswith((".txt", ".md")):
                full_path = os.path.join(source_path, filename)
                all_source_files.append(os.path.normpath(full_path))
    
    new_files = [f for f in all_source_files if f not in processed_files]

    if not new_files:
        print(f"Source '{source_name}' is already up-to-date. Nothing to do.")
        return

    print(f"Found {len(new_files)} new document(s) for '{source_name}'.")

    # 2. Load and chunk ONLY the new documents
    new_docs = []
    for file_path in new_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                new_docs.append(Document(page_content=f.read(), metadata={"source": os.path.basename(file_path)}))
        except Exception as e:
            print(f"  - Error reading {file_path}: {e}")

    if not new_docs:
        print("Could not read new files. Halting for this source.")
        return
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunked_docs = text_splitter.split_documents(new_docs)
    print(f"Split new documents into {len(chunked_docs)} chunks.")

    # 3. Embed and add to the index
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    
    if os.path.exists(index_path):
        print(f"Loading existing index '{index_path}' to update...")
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        print("Adding new documents to existing index...")
    else:
        print(f"No index found at '{index_path}'. Creating a new one...")
        # Create the index with the first batch
        vector_store = FAISS.from_documents(documents=chunked_docs[:50], embedding=embeddings)
        chunked_docs = chunked_docs[50:] # Remove the processed batch
        print("  - Initial batch processed...")

    # Process remaining chunks in batches for both new and updated indexes
    if chunked_docs:
        for i in range(0, len(chunked_docs), 50):
            batch = chunked_docs[i:i + 50]
            vector_store.add_documents(documents=batch)
            if len(chunked_docs) > 50:
                 print(f"  - Processed batch {i // 50 + 1}... Pausing for 60 seconds...")
                 time.sleep(60)

    # 4. Save index and update log
    print(f"Saving updated index to '{index_path}'...")
    vector_store.save_local(index_path)
    update_processed_log(new_files)
    print(f"--- FINISHED PROCESSING: {source_name.upper()} ---")

def main():
    if not os.getenv("GOOGLE_API_KEY"):
        print("FATAL ERROR: GOOGLE_API_KEY is not set.")
        return

    processed_files = get_processed_files()
    for source_name, source_info in KNOWLEDGE_SOURCES.items():
        process_single_source(source_name, source_info, processed_files)

    print("\nAll knowledge sources are now up to date.")

if __name__ == "__main__":
    main()