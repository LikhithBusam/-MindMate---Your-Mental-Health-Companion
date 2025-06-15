import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set up Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize session state for vectorstore
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# Load PDF and create vector store
def load_knowledge():
    with st.spinner("🧠 Loading and indexing the document..."):
        loader = PyPDFLoader("mental_health_Document.pdf")
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        st.session_state.vectorstore = vectorstore
        st.success("✅ Knowledge base ready!")

# Set up QA chain
def get_qa_chain():
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-8b-8192",
        temperature=0.7
    )

    prompt_template = """
    Use the following context from the mental health document to answer the question.
    If you don't know the answer, say so clearly.
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain_type_kwargs = {"prompt": PROMPT}
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )
    return qa

# Streamlit UI
st.set_page_config(page_title="MindMate - Mental Health Companion", layout="centered")
st.title("🧠 MindMate - Your Mental Health Companion")
st.markdown("Talk to me anytime — I'm here to listen, help, and guide.")

# Load document button
if st.button("📂 Load Mental Health Document"):
    load_knowledge()

# Chat interface
if st.session_state.vectorstore:
    user_input = st.text_input("Ask a question about mental health:", "")
    
    if user_input:
        qa_chain = get_qa_chain()
        result = qa_chain.invoke({"query": user_input})
        
        st.markdown(f"**You:** {user_input}")
        st.markdown(f"**MindMate:** {result['result']}")
        
        with st.expander("📚 Show sources"):
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Source {i+1}:**")
                st.write(doc.page_content[:400] + "...")

else:
    st.warning("⚠️ Please load the document first using the button above.")