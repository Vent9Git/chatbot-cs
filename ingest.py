import os
os.environ["ANONYMIZED_TELEMETRY"] = "False" 
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1" 

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Define Data & Database Paths
DATA_PATH = "data/customer_support.csv"
CHROMA_PATH = "chroma_db"

def create_vector_db():
    print(f"1. Reading data from {DATA_PATH}...")
    # If your CSV file uses a delimiter other than a comma (for example semicolon),
    # you can add: csv_args={'delimiter': ';'}
    loader = CSVLoader(file_path=DATA_PATH, encoding="utf-8")
    documents = loader.load()

    print(f"   -> Total raw data: {len(documents)} rows.")
    # documents = documents[:1000]  # ONLY USE THE FIRST 200 ROWS!

    print(f"   -> Successfully loaded {len(documents)} rows of data.\n")

    print("2. Splitting text (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   -> Data split into {len(chunks)} chunks.\n")

    print("3. Downloading/Loading AI Embedding model...")
    # BAAI/bge-small-en-v1.5 is fast and works well for English/general text
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    print("4. Creating ChromaDB Vector Database...")
    # Remove old DB if it already exists (optional, avoids duplicates on repeated runs)
    if os.path.exists(CHROMA_PATH):
        print("   -> Overwriting existing database...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    print(f"\n✅ STAGE 1 COMPLETE! Database saved in '{CHROMA_PATH}'.")

if __name__ == "__main__":
    create_vector_db()