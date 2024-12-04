from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import openai
import langchain

# Disable verbose logging from Langchain
langchain.verbose = False

# Load environment variables
load_dotenv()


# Function to extract text from uploaded PDFs
def process_uploaded_files(uploaded_files):
    all_text = ""
    for file in uploaded_files:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        all_text += text
    return all_text


# Function to generate embeddings for the extracted text
def generate_embeddings(text):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

    # Create chunks of text for embedding
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # Create FAISS index for storing embeddings
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base


def main():
    st.title("Documents Question Answering")
    st.write("Upload multiple PDF or text files to get started.")

    # Sidebar for file upload
    uploaded_files = st.sidebar.file_uploader("Upload your documents", type=["pdf", "txt"], accept_multiple_files=True)

    knowledge_base = None

    if uploaded_files:
        st.sidebar.write(f"Uploaded {len(uploaded_files)} file(s).")

        # Process the uploaded files
        all_text = process_uploaded_files(uploaded_files)

        if all_text:
            st.write("Extracted text from your documents:")
            st.write(all_text[:1000])  # Display first 1000 characters for preview

            # Generate embeddings and store them in FAISS
            knowledge_base = generate_embeddings(all_text)

            st.write("Embeddings generated and indexed for question answering.")
        else:
            st.write("No text found in the uploaded files.")

    # Section for asking questions to the documents
    query = st.text_input('Ask a question about the documents:')

    if query:
        if knowledge_base:
            docs = knowledge_base.similarity_search(query)

            # Create a chain for question answering
            llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
            chain = load_qa_chain(llm, chain_type="stuff")

            with get_openai_callback() as cost:
                response = chain.invoke(input={"question": query, "input_documents": docs})
                print(cost)

                # Display the response from the LLM
                st.write(response["output_text"])
        else:
            st.write("Please upload documents to ask questions.")


if __name__ == "__main__":
    main()
