import os
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found in environment")

def setup_qa_system(file_path: str):
    """
    Sets up a Retrieval-Augmented Generation (RAG) system
    using Gemini API for question-answering over a given PDF.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    index_path = "faiss_index"

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    # Check for saved FAISS index
    if os.path.exists(index_path):
        print("Loading existing FAISS index...")
        vector_store = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("Creating new FAISS index...")
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)
        print(f"Split into {len(chunks)} chunks.")

        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(index_path)
        print("Saved FAISS index for future use.")

    retriever = vector_store.as_retriever()

    # Initialize Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0,
        google_api_key=GOOGLE_API_KEY
    )

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    return qa_chain


def main():
    """
    Console-based testing for QA functionality.
    
    """
    try:
        qa_chain = setup_qa_system(r"D:\Coding\third\peice.pdf")
    except Exception as e:
        print("Error:", e)
        return

    print(" Gemini-based QA ready. Type questions ('exit' to quit).")

    while True:
        try:
            question = input("\nAsk a question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if question.lower() == "exit":
            break
        if not question:
            continue

        try:
            answer = qa_chain.run(question)
            print("\nAnswer:\n", answer)
        except Exception as e:
            print("Error obtaining answer:", e)


if __name__ == "__main__":
    main()
