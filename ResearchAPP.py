import os
import textract
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import faiss
import streamlit as st

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "fe51611c5a00406baa1d7325c8684d87"

# Set up OpenAI API configuration
openai.api_key = "fe51611c5a00406baa1d7325c8684d87"
openai.api_type = "azure"
openai.api_base = "https://iocl-oai.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
engine = "gpt_35_turbo_16k"

# Create a Streamlit app
def main():
    st.title("PDF Chatbot")
    st.write("Upload PDF files and ask questions to get answers.")

    # Upload PDF files
    pdf_files = st.file_uploader("Upload PDF Files", type=['pdf'], accept_multiple_files=True)

    # Initialize chatbot variables
    chat_history = []
    qa = None

    if pdf_files:
        text = ""
        for pdf_file in pdf_files:
            doc = textract.process(pdf_file)
            text += doc.decode('utf-8')

        # Token counting function
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        def count_tokens(text: str) -> int:
            return len(tokenizer.encode(text))

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=50,
            length_function=count_tokens,
        )

        chunks = text_splitter.create_documents([text])

        # Create a list of token counts
        token_counts = [count_tokens(chunk.page_content) for chunk in chunks]

        # Create a DataFrame from the token counts
        df = pd.DataFrame({'Token Count': token_counts})

        # Create embeddings
        embeddings = OpenAIEmbeddings()

        # Create a vector database
        db = FAISS.from_documents(chunks, embeddings)

        # Create conversation chain
        qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.1, engine=engine), db.as_retriever())

    st.write("Welcome to the PDF Chatbot! Type 'exit' to stop.")
    input_box = st.text_input("Please enter your question:")

    if st.button("Ask"):
        if input_box.lower() == 'exit':
            st.write("Thank you for using the PDF Chatbot!")
        else:
            result = qa({"question": input_box, "chat_history": chat_history})
            chat_history.append((input_box, result['answer']))
            st.write("User: ", input_box)
            st.write("Chatbot: ", result['answer'])

if __name__ == "__main__":
    main()
