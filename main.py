# impotr libraries
import os
os.environ["PWD"] = os.getcwd()
    
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_ollama import OllamaLLM, OllamaEmbeddings

from transformers import pipeline
import streamlit as st


# SET TITLE OF PAGE
st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄")
st.title("PDF chatbot (Local RAG)")
st.write("Upload a PDFs and ask questions about its content.")


# CREATE A FOLDER TO STORE PDFS
UPLOAD_DIR = "upload_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok = True)


# UPLOAD PDFs
uploaded_files = st.file_uploader("Upload pdf files", type = "pdf", accept_multiple_files = True)


# only continue if user upload a file
if uploaded_files is not None:
    # pypdfloader needs a file path, so we just share it with locally first
    # save and load pdf
    all_documents = []
    for upload_file in uploaded_files:
        pdf_path = os.path.join(UPLOAD_DIR, upload_file.name)
        with open(pdf_path, "wb") as f:
            f.write(upload_file.read())
        
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()

        # Add filename metadata to each page
        for doc in documents:
            doc.metadata["source_file"] = upload_file.name
        all_documents.extend(documents)

    st.success(f"{len(uploaded_files)} PDF(s) uploaded success")
    st.write(f"Loaded {len(all_documents)} Page(s) from the PDF.")




    # SPLIT PDF TEXT IN CHUNKS
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1200,
        chunk_overlap = 200
    )
    chunk = text_splitter.split_documents(all_documents)
    st.write(f"Created {len(chunk)} text chunk(s).")




    # CREATE EMBEDDINZG MODEL
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )



    
    # STORE CHUNK IN VECTOR DB USING FAISS
    vectorstore  = FAISS.from_documents(chunk, embeddings)
    




    # CREATE RETRIEVER - WHICH RETRIEVE MOST RELEVENT CHUNKS
    retriever = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k": 4} #return top 3 most relevant chunks
    )




    # CREATE LOCAL LLM TO GENERATE POSSIBLE TEXT RESPONSE FOR ANSWER
    llm = OllamaLLM(model="llama3")


    # CREATE PROMPT
    # -------------------------------
    prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant answering questions only from the provided context.

        Rules:
        1. Use only the provided context.
        2. If the answer is not clearly in the context, say: "I do not know based on the provided documents."
        3. If multiple documents are relevant, summarize them clearly.
        4. Do not invent facts.
        5. When possible, keep the answer concise and grounded.

        Context:
        {context}

        Question:
        {input}

        Answer:
        """)
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    # QUESTION INPUT
    question = st.text_input("Ask a question about the PDF:")

    if question:
        with st.spinner("Shanti Rakh Loda..."):
            # RUN FULL RAG PIPELINE
            response = rag_chain.invoke({"input": question})

        st.subheader("Answer")
        st.write(response["answer"])

        # optional debug section
        with st.expander("Retrieved Chunks"):
            for i, doc in enumerate(response["context"], start=1):
                source_file = doc.metadata.get("source_file", "Unknown file")
                page_num = doc.metadata.get("page", 0) + 1

                st.markdown(f"**Chunk {i}**")
                st.markdown(f"**Source File:** {source_file}")
                st.markdown(f"**Page:** {page_num}")
                st.write(doc.page_content[:1000])
                st.markdown("---")