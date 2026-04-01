import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="CS AI Assistant", page_icon="🤖")
st.title("🤖 Customer Support Assistant")
st.caption("Ditenagai oleh RAG, Groq, dan Streamlit")

# --- Memuat Database dan AI (Gunakan Cache agar efisien) ---
# @st.cache_resource memastikan model & database hanya diload 1x saat aplikasi menyala
@st.cache_resource
def load_rag_chain():
    print("Memuat Vector Database & LLM...")
    # 1. Load Embeddings & Database Chroma
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2. Load LLM Groq (Mengambil API Key dari Secrets Streamlit agar aman!)
    api_key = st.secrets["GROQ_API_KEY"]
    llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.1-8b-instant", 
        groq_api_key=api_key
    )

    # 3. Gabungkan menjadi Conversational Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# Panggil fungsi untuk memuat RAG
qa_chain = load_rag_chain()

# --- Inisialisasi Memori Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Histori khusus untuk format LangChain
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Menampilkan histori chat di layar ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Kolom Input Chat ---
if prompt := st.chat_input("Ketik pertanyaan Anda di sini..."):
    # Tampilkan pesan user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Siapkan wadah untuk AI
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ *Sedang berpikir...*")
        
        try:
            # Eksekusi RAG langsung dari dalam file yang sama!
            hasil = qa_chain.invoke({
                "question": prompt,
                "chat_history": st.session_state.chat_history
            })
            
            jawaban_ai = hasil["answer"]
            sumber = [doc.page_content for doc in hasil["source_documents"]]
            
            # Update UI dengan jawaban
            message_placeholder.markdown(jawaban_ai)
            with st.expander("Tampilkan Sumber Dokumen RAG"):
                for i, doc in enumerate(sumber):
                    st.info(f"**Dokumen {i+1}:**\n{doc}")
            
            # Simpan ke histori
            st.session_state.messages.append({"role": "assistant", "content": jawaban_ai})
            st.session_state.chat_history.append((prompt, jawaban_ai))
            
        except Exception as e:
            message_placeholder.error(f"Terjadi kesalahan: {e}")