import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain.schema import HumanMessage
import os

# Load environment variables
load_dotenv()

# Set your Gemini API key
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize session state variables
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'email' not in st.session_state:
    st.session_state.email = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Login function allowing any Gmail email and any password
def login():
    st.sidebar.title("🔐 Login")
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")
    login_button = st.sidebar.button("Login")

    if login_button:
        if email.endswith("@gmail.com") and password:
            st.session_state.logged_in = True
            st.session_state.email = email
            st.sidebar.success(f"Logged in as {email}")
        else:
            st.sidebar.error("❌ Please enter a valid Gmail address and password")

# Logout button in sidebar
def logout():
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.email = ""
        st.session_state.vectorstore = None
        st.session_state.chat_history = []
        st.experimental_rerun()

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
    You are MindMate, a compassionate assistant helping users understand mental health topics.
    Use the following context from the mental health document to answer the question.
    If no relevant info exists, respond with empathy and general guidance.

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

# Fallback response when no knowledge found
def fallback_response(user_input):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.4,
        max_tokens=1024,
        google_api_key=api_key
    )

    prompt = f"""
    A user asked: "{user_input}"
    
    Provide a supportive and empathetic response related to mental health.
    Make sure to emphasize that this is not medical advice and suggest reaching out to professionals if needed.
    Keep the tone warm and comforting.
    """

    message = HumanMessage(content=prompt)
    result = llm.invoke([message])
    return result.content

# --- Main app ---

st.set_page_config(page_title="MindMate - Mental Health Companion", layout="centered")

if not st.session_state.logged_in:
    login()
    st.info("⚠️ Please log in to use MindMate.")
else:
    logout()
    st.title("🧠 MindMate - Your Mental Health Companion")
    st.markdown("Talk to me anytime — I'm here to listen, help, and guide.")

    st.markdown("""
    📌 *Disclaimer*:  
    I am an AI companion and cannot replace professional mental health care.  
    If you're feeling overwhelmed or in crisis, please reach out to a mental health professional or call a helpline.
    """)

    # Load document button
    if st.button("📂 Load Mental Health Document"):
        load_knowledge()

    # Chat interface
    if st.session_state.vectorstore:
        user_input = st.text_input("💬 Ask a question about mental health:", "")

        if user_input:
            with st.spinner("🔍 Thinking..."):
                try:
                    qa_chain = get_qa_chain()
                    result = qa_chain.invoke({"query": user_input})

                    answer = result['result']
                    sources = result.get('source_documents', [])

                    # Handle empty or unhelpful answers
                    if "does not offer" in answer.lower() or "i'm sorry" in answer.lower():
                        answer = fallback_response(user_input)

                    # Save to chat history
                    st.session_state.chat_history.append(("You", user_input))
                    st.session_state.chat_history.append(("MindMate", answer))

                    # Display chat
                    for speaker, msg in st.session_state.chat_history:
                        if speaker == "You":
                            st.markdown(f"**🧑‍💻 You:** {msg}")
                        else:
                            st.markdown(f"**🧠 MindMate:** {msg}")

                    if sources:
                        with st.expander("📚 Show sources"):
                            for i, doc in enumerate(sources):
                                st.markdown(f"**Source {i+1}:**")
                                st.write(doc.page_content[:400] + "...")

                except Exception as e:
                    st.error("⚠️ Something went wrong. Please try again.")
                    st.write(str(e))
    else:
        st.warning("⚠️ Please load the document first using the button above.")

