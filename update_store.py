# --- update_store.py ---
# This script intelligently builds OR incrementally updates your knowledge base.
# It only processes new files, saving significant time and API calls.

import os
from dotenv import load_dotenv
import time

# --- LangChain Imports ---
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# This is where the final database will be saved/updated.
FAISS_INDEX_PATH = "faiss_index" 
# This log file tracks which files have already been processed.
PROCESSED_LOG_PATH = "processed_files.log"
# Add any new directories you want to scan here.
SEARCH_DIRECTORIES = [".", "dossiers", "essays"]
# Using the newer, more powerful embedding model.
EMBEDDING_MODEL = "models/text-embedding-004"

def get_processed_files():
    """Reads the log file and returns a set of processed file paths."""
    if not os.path.exists(PROCESSED_LOG_PATH):
        return set()
    with open(PROCESSED_LOG_PATH, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f)

def update_processed_files(file_paths):
    """Appends a list of newly processed files to the log."""
    with open(PROCESSED_LOG_PATH, 'a', encoding='utf-8') as f:
        for path in file_paths:
            f.write(path + '\n')

def update_vector_store():
    """
    Main function to build (if new) or update (if existing) the vector store.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        print("FATAL ERROR: GOOGLE_API_KEY is not set in the .env file.")
        return

    print("--- Starting Knowledge Base Update Process ---")

    # 1. Find all potential documents and check against the processed log.
    print("Step 1: Scanning for new documents...")
    processed_files = get_processed_files()
    all_found_files = []
    for directory in SEARCH_DIRECTORIES:
        if not os.path.isdir(directory):
            continue
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.endswith(".txt"):
                    all_found_files.append(os.path.join(root, filename))

    new_files_to_process = [f for f in all_found_files if f not in processed_files]

    if not new_files_to_process:
        print("Knowledge base is already up to date. No new documents found. Halting.")
        return

    print(f"Found {len(new_files_to_process)} new document(s) to add.")

    # 2. Load the new documents' content.
    print("\nStep 2: Loading content from new documents...")
    new_docs = []
    for file_path in new_files_to_process:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                # Using the relative path for cleaner metadata
                relative_path = os.path.relpath(file_path)
                doc = Document(page_content=text, metadata={"source": relative_path})
                new_docs.append(doc)
        except Exception as e:
            print(f"    - Error reading {file_path}: {e}")

    if not new_docs:
        print("Error: Could not load content from new files. Halting.")
        return
        
    # 3. Split ONLY the new documents into smaller chunks.
    print("\nStep 3: Splitting new documents into manageable chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunked_docs = text_splitter.split_documents(new_docs)
    print(f"Split new documents into {len(chunked_docs)} chunks.")

    # 4. Initialize the embeddings model.
    print(f"\nStep 4: Initializing Google Embeddings model ({EMBEDDING_MODEL})...")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    # 5. Load existing index or create a new one.
    if os.path.exists(FAISS_INDEX_PATH):
        # --- UPDATE PATH ---
        print("\nStep 5: Loading existing FAISS index to update...")
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        
        print(f"Adding {len(chunked_docs)} new chunks to the existing index...")
        # Add the new documents in batches to respect rate limits
        for i in range(0, len(chunked_docs), 50):
            batch = chunked_docs[i:i + 50]
            vector_store.add_documents(documents=batch)
            if len(chunked_docs) > 50:
                 print(f"  - Processed batch {i // 50 + 1}... Pausing for 60 seconds...")
                 time.sleep(60)

    else:
        # --- INITIAL BUILD PATH ---
        print("\nStep 5: No existing index found. Creating new FAISS index...")
        print("This will make many API calls and may take several minutes.")
        
        # Create the index from scratch in batches
        vector_store = FAISS.from_documents(documents=chunked_docs[:50], embedding=embeddings)
        print("  - Initial batch processed...")
        for i in range(50, len(chunked_docs), 50):
            batch = chunked_docs[i:i + 50]
            vector_store.add_documents(documents=batch)
            print(f"  - Processed batch {i // 50 + 1}... Pausing for 60 seconds...")
            time.sleep(60)
            
    # 6. Save the updated index and log the newly processed files.
    print("\nStep 6: Saving updated index to disk...")
    vector_store.save_local(FAISS_INDEX_PATH)
    
    print("Updating the log of processed files...")
    update_processed_files(new_files_to_process)
    
    print("\n--- Update Complete! ---")
    print(f"Knowledge base '{FAISS_INDEX_PATH}' is now up to date.")


if __name__ == "__main__":
    update_vector_store()