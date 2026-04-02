# 🤖 Customer Support AI Assistant (RAG)

An end-to-end AI-powered Customer Support Chatbot built with **Retrieval-Augmented Generation (RAG)**.  
This application acts as a smart virtual assistant capable of answering user queries based on a massive, locally embedded knowledge base of **25,000+ customer support operational records**.

It features conversational memory, ultra-fast inference via Groq API, and an interactive UI built with Streamlit.

---

## ✨ Key Features

- 🔍 **Retrieval-Augmented Generation (RAG)**  
  Provides grounded and accurate responses based strictly on company SOPs and datasets.

- 📚 **Massive Knowledge Base**  
  Powered by a locally persisted ChromaDB containing **25,000+ embedded documents**.

- 🧠 **Conversational Memory**  
  Maintains chat context for better follow-up handling using `ConversationalRetrievalChain`.

- ⚡ **Ultra-Fast Inference**  
  Uses `Llama-3.1-8b-instant` via Groq API for near real-time responses.

- 💻 **Seamless UI/UX**  
  Clean and responsive interface built with Streamlit.

---

## 🛠️ Tech Stack

| Component        | Technology |
|-----------------|-----------|
| Frontend & UI   | Streamlit |
| Orchestration   | LangChain (`langchain-classic`, `langchain-community`) |
| LLM             | Llama 3.1 (Groq API) |
| Embeddings      | BAAI/bge-small-en-v1.5 (HuggingFace) |
| Vector Database | ChromaDB (local persistent storage) |

---

## 📂 Project Structure
```text
chatbot-cs/
├── chroma_db/            # Pre-computed vector database (25k+ records)
├── .streamlit/           # Local secrets directory (Ignored by Git)
│   └── secrets.toml      # Contains GROQ_API_KEY for local run
├── main.py               # Main application script (Streamlit UI & RAG logic)
├── requirements.txt      # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # Project documentation
```

## 🏗️ How to Build or Update the Vector Database (ChromaDB)

This repository includes a lightweight version of ChromaDB for quick testing. However, if you want to ingest the full 25,000+ dataset or use your own custom data, you can easily rebuild the database using the provided `ingest.py` script.

### 1. Ingesting the Full Dataset
1. Prepare your data in a CSV format and place it inside the `data/` directory.
2. Open `ingest.py` and adjust the `DATA_PATH` variable to match your specific file name:
   ```python
   # 1. Define Data & Database Paths
   DATA_PATH = "data/customer_support.csv" # Change this if your file name is different
   CHROMA_PATH = "chroma_db"
   ```

##  🚀 How to Run Locally
### 1. Clone Repository
```bash
git clone https://github.com/UsernameAnda/chatbot-rag-jatis.git
cd chatbot-rag-jatis
```

### 2. Set Up Virtual Environment
It is recommended to use a virtual environment to avoid dependency conflicts.
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables (API Key)
Create a .streamlit folder and a secrets.toml file inside it to securely store your Groq API Key.
```bash
mkdir .streamlit
touch .streamlit/secrets.toml
```

Open .streamlit/secrets.toml and add your Groq API key:
```bash
GROQ_API_KEY = "your_groq_api_key_here"
```

### 5. Run the Application
```bash
streamlit run main.py
```
The application will be accessible at http://localhost:8501.

## 🌐 Deployment (Streamlit Community Cloud)
This app is optimized for single-tier deployment on Streamlit Community Cloud.

1. Push your repository to GitHub (ensure .streamlit/ and venv/ are in .gitignore). Important: Make sure the chroma_db/ folder is pushed so the cloud server has access to the pre-computed embeddings.
2. Go to [Streamlit Share](https://share.streamlit.io/) and log in with your GitHub account.
3. Click "New app" and select your repository, branch, and main.py as the main file path.
4. Click on "Advanced settings..." before deploying.
5. In the Secrets field, paste your Groq API key:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```
6. Click Deploy!

## 📊 Dataset Acknowledgment
The customer support dataset used to build the ChromaDB vector index in this project was sourced from [Aditya Kumar's RAG Chatbot Repository](https://github.com/30adityakumar/rag-chatbot). 

While the dataset originates from the aforementioned repository, the core architecture of this application has been completely re-engineered to utilize a different tech stack, migrating from FAISS/OpenAI to a persistent **ChromaDB**, **LangChain**, and open-source **Llama 3.1 via Groq API** for ultra-fast inference.