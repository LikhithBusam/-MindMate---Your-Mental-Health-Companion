import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set your Gemini API key
api_key = os.getenv("GOOGLE_API_KEY")  # replace this with your actual Gemini API key

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

# Set up QA chain using Gemini
def get_qa_chain():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        max_tokens=1024,
        google_api_key=api_key
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

# Streamlit UI setup
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

        # Check if the result is empty or not helpful
        if "does not offer" in result['result'].lower() or "i'm sorry" in result['result'].lower():
            st.info("ℹ️ The document didn't contain info on that. Here's a general response:")

            fallback_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.4,
                max_tokens=1024,
                google_api_key=api_key
            )

            fallback_response = fallback_llm.invoke(
                f"{user_input} (Answer this in an empathetic and helpful way for someone dealing with mental health issues)"
            )
            st.markdown(f"**MindMate (General Advice):** {fallback_response.content}")
        else:
            st.markdown(f"**MindMate:** {result['result']}")

            with st.expander("📚 Show sources"):
                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Source {i+1}:**")
                    st.write(doc.page_content[:400] + "...")
else:
    st.warning("⚠️ Please load the document first using the button above.")
