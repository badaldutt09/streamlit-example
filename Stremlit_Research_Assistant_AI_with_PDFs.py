{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/badaldutt09/streamlit-example/blob/master/Stremlit_Research_Assistant_AI_with_PDFs.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 0. Installs, Imports and API Keys"
      ],
      "metadata": {
        "id": "Q24Y-g6h-Bg0"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# RUN THIS CELL FIRST!\n",
        "!pip install -q langchain==0.0.150 pypdf pandas matplotlib tiktoken textract transformers openai faiss-cpu"
      ],
      "metadata": {
        "id": "gk2J2sYYjTkM"
      },
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install openai"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "JGhVPZceOqGI",
        "outputId": "8f67523b-7370-4f99-a9ea-67299ebf7711"
      },
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: openai in /usr/local/lib/python3.10/dist-packages (0.28.1)\n",
            "Requirement already satisfied: requests>=2.20 in /usr/local/lib/python3.10/dist-packages (from openai) (2.31.0)\n",
            "Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from openai) (4.66.1)\n",
            "Requirement already satisfied: aiohttp in /usr/local/lib/python3.10/dist-packages (from openai) (3.8.6)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.20->openai) (3.3.0)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.20->openai) (3.4)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.20->openai) (2.0.7)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.20->openai) (2023.7.22)\n",
            "Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->openai) (23.1.0)\n",
            "Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp->openai) (6.0.4)\n",
            "Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /usr/local/lib/python3.10/dist-packages (from aiohttp->openai) (4.0.3)\n",
            "Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->openai) (1.9.2)\n",
            "Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp->openai) (1.4.0)\n",
            "Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp->openai) (1.3.1)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pip install streamlit"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "IhfDfQl7aa36",
        "outputId": "c9b5d0dc-cd6e-4efe-e22e-fc8782ac60a8"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: streamlit in /usr/local/lib/python3.10/dist-packages (1.27.2)\n",
            "Requirement already satisfied: altair<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.2.2)\n",
            "Requirement already satisfied: blinker<2,>=1.0.0 in /usr/lib/python3/dist-packages (from streamlit) (1.4)\n",
            "Requirement already satisfied: cachetools<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (5.3.1)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.1.7)\n",
            "Requirement already satisfied: importlib-metadata<7,>=1.4 in /usr/local/lib/python3.10/dist-packages (from streamlit) (6.8.0)\n",
            "Requirement already satisfied: numpy<2,>=1.19.3 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.23.5)\n",
            "Requirement already satisfied: packaging<24,>=16.8 in /usr/local/lib/python3.10/dist-packages (from streamlit) (23.2)\n",
            "Requirement already satisfied: pandas<3,>=1.3.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.5.3)\n",
            "Requirement already satisfied: pillow<11,>=7.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (9.4.0)\n",
            "Requirement already satisfied: protobuf<5,>=3.20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (3.20.3)\n",
            "Requirement already satisfied: pyarrow>=6.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (9.0.0)\n",
            "Requirement already satisfied: python-dateutil<3,>=2.7.3 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.8.2)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.31.0)\n",
            "Requirement already satisfied: rich<14,>=10.14.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (13.6.0)\n",
            "Requirement already satisfied: tenacity<9,>=8.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.2.3)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.3.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.5.0)\n",
            "Requirement already satisfied: tzlocal<6,>=1.1 in /usr/local/lib/python3.10/dist-packages (from streamlit) (5.1)\n",
            "Requirement already satisfied: validators<1,>=0.2 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.22.0)\n",
            "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /usr/local/lib/python3.10/dist-packages (from streamlit) (3.1.40)\n",
            "Requirement already satisfied: pydeck<1,>=0.8.0b4 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.8.1b0)\n",
            "Requirement already satisfied: tornado<7,>=6.0.3 in /usr/local/lib/python3.10/dist-packages (from streamlit) (6.3.2)\n",
            "Requirement already satisfied: watchdog>=2.1.5 in /usr/local/lib/python3.10/dist-packages (from streamlit) (3.0.0)\n",
            "Requirement already satisfied: entrypoints in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.4)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (3.1.2)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (4.19.1)\n",
            "Requirement already satisfied: toolz in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.12.0)\n",
            "Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.10/dist-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.11)\n",
            "Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.10/dist-packages (from importlib-metadata<7,>=1.4->streamlit) (3.17.0)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.3.0->streamlit) (2023.3.post1)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil<3,>=2.7.3->streamlit) (1.12.0)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.3.0)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.4)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2.0.7)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2023.7.22)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (2.16.1)\n",
            "Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.10/dist-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.1)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.3)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (23.1.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2023.7.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.30.2)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.10.6)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.2)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "import streamlit as st\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "from transformers import GPT2TokenizerFast\n",
        "from langchain.document_loaders import PyPDFLoader\n",
        "from langchain.text_splitter import RecursiveCharacterTextSplitter\n",
        "from langchain.embeddings import OpenAIEmbeddings\n",
        "from langchain.vectorstores import FAISS\n",
        "from langchain.chains.question_answering import load_qa_chain\n",
        "from langchain.llms import OpenAI\n",
        "from langchain.chat_models import ChatOpenAI\n",
        "from langchain.chains import ConversationalRetrievalChain"
      ],
      "metadata": {
        "id": "l-uszlwN641q"
      },
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "os.environ[\"OPENAI_API_KEY\"] = \"fe51611c5a00406baa1d7325c8684d87\""
      ],
      "metadata": {
        "id": "E2Buv5Y0uFr8"
      },
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#importing open ai resource\n",
        "import openai\n",
        "import json\n",
        "# Set OpenAI API configuration\n",
        "openai.api_key = \"fe51611c5a00406baa1d7325c8684d87\"\n",
        "openai.api_type =\"azure\"\n",
        "openai.api_base = \"https://iocl-oai.openai.azure.com/\"\n",
        "openai.api_version =\"2023-07-01-preview\"\n",
        "engine=\"gpt_35_turbo_16k\""
      ],
      "metadata": {
        "id": "vtNofmMSd-jr"
      },
      "execution_count": 17,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 1. Loading PDFs and chunking with LangChain"
      ],
      "metadata": {
        "id": "RLULMPXa-Hu8"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Define the Streamlit app\n",
        "def main():\n",
        "    st.title(\"PDF Chatbot\")\n",
        "    st.write(\"Upload PDF files and ask questions to get answers.\")\n",
        "\n",
        "    # Add code to upload PDF files\n",
        "    pdf_files = st.file_uploader(\"Upload PDF Files\", type=['pdf'], accept_multiple_files=True)\n",
        "\n",
        "    # Process uploaded PDFs and create the chatbot\n",
        "    if pdf_files:\n",
        "        text = \"\"\n",
        "        for pdf_file in pdf_files:\n",
        "            doc = textract.process(pdf_file)\n",
        "            text += doc.decode('utf-8')\n",
        "\n",
        "\n",
        "        # Token counting function\n",
        "        tokenizer = GPT2TokenizerFast.from_pretrained(\"gpt2\")\n",
        "        def count_tokens(text: str) -> int:\n",
        "            return len(tokenizer.encode(text))\n",
        "\n",
        "        # Split text into chunks\n",
        "        text_splitter = RecursiveCharacterTextSplitter(\n",
        "            chunk_size=1500,\n",
        "            chunk_overlap=50,\n",
        "            length_function=count_tokens,\n",
        "        )\n",
        "\n",
        "        chunks = text_splitter.create_documents([text])\n",
        "\n",
        "        # Create a list of token counts\n",
        "        token_counts = [count_tokens(chunk.page_content) for chunk in chunks]\n",
        "\n",
        "        # Create a DataFrame from the token counts\n",
        "        df = pd.DataFrame({'Token Count': token_counts})\n",
        "\n",
        "        # Create embeddings\n",
        "        embeddings = OpenAIEmbeddings()\n",
        "\n",
        "        # Create a vector database\n",
        "        db = FAISS.from_documents(chunks, embeddings)\n",
        "\n",
        "        # Create conversation chain\n",
        "        qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.1, engine=engine), db.as_retriever())\n",
        "\n",
        "    st.write(\"Welcome to the PDF Chatbot! Type 'exit' to stop.\")\n",
        "    input_box = st.text_input(\"Please enter your question:\")\n",
        "\n",
        "    if st.button(\"Ask\"):\n",
        "        if input_box.lower() == 'exit':\n",
        "            st.write(\"Thank you for using the PDF Chatbot!\")\n",
        "        else:\n",
        "            result = qa({\"question\": input_box, \"chat_history\": chat_history})\n",
        "            chat_history.append((input_box, result['answer']))\n",
        "            st.write(\"User: \", input_box)\n",
        "            st.write(\"Chatbot: \", result['answer'])\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    main()"
      ],
      "metadata": {
        "id": "lE7lLsSPa1qZ"
      },
      "execution_count": 18,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!streamlit run app.py"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5QCs39ItdNqG",
        "outputId": "bf98160f-03ce-49bd-918f-5e374bcd7122"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Collecting usage statistics. To deactivate, set browser.gatherUsageStats to False.\n",
            "\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m\u001b[1m  You can now view your Streamlit app in your browser.\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Network URL: \u001b[0m\u001b[1mhttp://172.28.0.12:8501\u001b[0m\n",
            "\u001b[34m  External URL: \u001b[0m\u001b[1mhttp://35.245.73.175:8501\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Stopping...\u001b[0m\n",
            "^C\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Advanced method - Split by chunk\n",
        "\n",
        "# Step 1: Convert PDF to text\n",
        "import textract\n",
        "\n",
        "# List of PDF file paths\n",
        "pdf_files = [\"/content/Andaman Sea - Seismic Exploration De-Risking.pdf\", \"/content/SCOPE TOC-of-future-exploration-in-andaman-basin-jaipur-2015.pdf\"]\n",
        "\n",
        "# Iterate through each PDF and extract text\n",
        "text = \"\"\n",
        "for pdf_file in pdf_files:\n",
        "    doc = textract.process(pdf_file)\n",
        "    text += doc.decode('utf-8')\n",
        "\n",
        "# Step 2: Save the extracted text to a single .txt file\n",
        "with open('pdftext.txt', 'w', encoding='utf-8') as f:\n",
        "    f.write(text)\n",
        "\n",
        "# If you want to read the text from the file, you can do so:\n",
        "with open('pdftext.txt', 'r', encoding='utf-8') as f:\n",
        "    text = f.read()\n",
        "\n",
        "# Step 3: Create function to count tokens\n",
        "tokenizer = GPT2TokenizerFast.from_pretrained(\"gpt2\")\n",
        "\n",
        "def count_tokens(text: str) -> int:\n",
        "    return len(tokenizer.encode(text))\n",
        "\n",
        "# Step 4: Split text into chunks\n",
        "text_splitter = RecursiveCharacterTextSplitter(\n",
        "    # Set a really small chunk size, just to show.\n",
        "    chunk_size = 1500,\n",
        "    chunk_overlap  = 50,\n",
        "    length_function = count_tokens,\n",
        ")\n",
        "\n",
        "chunks = text_splitter.create_documents([text])"
      ],
      "metadata": {
        "id": "iADY2CXNlNq9",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 402
        },
        "outputId": "b9d8900c-8ded-48a6-db86-382e0eaa4017"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "error",
          "ename": "MissingFileError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mMissingFileError\u001b[0m                          Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-19-06ca46136520>\u001b[0m in \u001b[0;36m<cell line: 11>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     10\u001b[0m \u001b[0mtext\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m\"\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     11\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mpdf_file\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mpdf_files\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 12\u001b[0;31m     \u001b[0mdoc\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtextract\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mprocess\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpdf_file\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     13\u001b[0m     \u001b[0mtext\u001b[0m \u001b[0;34m+=\u001b[0m \u001b[0mdoc\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdecode\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'utf-8'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     14\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/textract/parsers/__init__.py\u001b[0m in \u001b[0;36mprocess\u001b[0;34m(filename, input_encoding, output_encoding, extension, **kwargs)\u001b[0m\n\u001b[1;32m     39\u001b[0m     \u001b[0;31m# make sure the filename exists\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     40\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mos\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpath\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mexists\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfilename\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 41\u001b[0;31m         \u001b[0;32mraise\u001b[0m \u001b[0mexceptions\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mMissingFileError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfilename\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     42\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     43\u001b[0m     \u001b[0;31m# get the filename extension, which is something like .docx for\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mMissingFileError\u001b[0m: The file \"/content/Andaman Sea - Seismic Exploration De-Risking.pdf\" can not be found.\nIs this the right path/to/file/you/want/to/extract.pdf?"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Result is many LangChain 'Documents' around 500 tokens or less (Recursive splitter sometimes allows more tokens to retain context)\n",
        "type(chunks[0])\n",
        "len(chunks)"
      ],
      "metadata": {
        "id": "KQ_gDkwep4q7"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Quick data visualization to ensure chunking was successful\n",
        "\n",
        "# Create a list of token counts\n",
        "token_counts = [count_tokens(chunk.page_content) for chunk in chunks]\n",
        "\n",
        "# Create a DataFrame from the token counts\n",
        "df = pd.DataFrame({'Token Count': token_counts})\n",
        "\n",
        "# Create a histogram of the token count distribution\n",
        "df.hist(bins=40, )\n",
        "\n",
        "# Show the plot\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "fK31bxDOpz1l"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 2. Embed text and store embeddings"
      ],
      "metadata": {
        "id": "_IlznUDK-i2m"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import faiss\n",
        "# Get embedding model\n",
        "embeddings = OpenAIEmbeddings()\n",
        "\n",
        "# Create vector database\n",
        "db = FAISS.from_documents(chunks, embeddings)"
      ],
      "metadata": {
        "id": "92ObhTAKnZzQ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "db"
      ],
      "metadata": {
        "id": "L4fAN6ZcFe-C"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 3. Setup retrieval function"
      ],
      "metadata": {
        "id": "2LPwdGDP-nPO"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Check similarity search is working\n",
        "query = \"What countries have oil and gas disocveries?\"\n",
        "docs = db.similarity_search(query)\n",
        "docs[0]"
      ],
      "metadata": {
        "id": "RWP92zGg5Nb_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Set OpenAI API configuration\n",
        "os.environ[\"OPENAI_API_KEY\"] = \"fe51611c5a00406baa1d7325c8684d87\"\n",
        "os.environ[\"Openai.Api_Type\"] =\"azure\"\n",
        "os.environ[\"Openai_Api_Base\"] = \"https://iocl-oai.openai.azure.com/\"\n",
        "os.environ[\"Openai_Api_Version\"] =\"2023-07-01-preview\"\n",
        "os.environ[\"Engine\"]=\"gpt_35_turbo_16k\""
      ],
      "metadata": {
        "id": "QYXduaNEHVkc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from langchain.chains.question_answering import load_qa_chain\n",
        "from langchain.llms import OpenAI\n"
      ],
      "metadata": {
        "id": "FfsrlK5BXoky"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 4. Integrating Similarity search with user Queries"
      ],
      "metadata": {
        "id": "9t68kdUPg9n9"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Create QA chain to integrate similarity search with user queries (answer query from knowledge base)\n",
        "\n",
        "engine=\"gpt_35_turbo_16k\"\n",
        "chain = load_qa_chain(ChatOpenAI(engine=engine, temperature=0.5, max_tokens=500),chain_type=\"stuff\")\n",
        "\n",
        "query = \"What countries have oil and gas disocveries?\"\n",
        "docs = db.similarity_search(query)\n",
        "\n",
        "chain.run(input_documents=docs, question=query)"
      ],
      "metadata": {
        "id": "AeQhg13ESxzC"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 5. Chatbot with chat memory"
      ],
      "metadata": {
        "id": "U_nH1qoL-w--"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from IPython.display import display\n",
        "import ipywidgets as widgets\n",
        "\n",
        "# Create conversation chain that uses our vectordb as retriver, this also allows for chat history management\n",
        "qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.1, engine=engine), db.as_retriever())"
      ],
      "metadata": {
        "id": "evF7_Dyhtcaf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "chat_history = []\n",
        "\n",
        "def on_submit(_):\n",
        "    query = input_box.value\n",
        "    input_box.value = \"\"\n",
        "\n",
        "    if query.lower() == 'exit':\n",
        "        print(\"Thank you for using the State of the Union chatbot!\")\n",
        "        return\n",
        "\n",
        "    result = qa({\"question\": query, \"chat_history\": chat_history})\n",
        "    chat_history.append((query, result['answer']))\n",
        "\n",
        "    display(widgets.HTML(f'<b>User:</b> {query}'))\n",
        "    display(widgets.HTML(f'<b><font color=\"blue\">Chatbot:</font></b> {result[\"answer\"]}'))\n",
        "\n",
        "print(\"Welcome to the Research Assitant chatbot! Type 'exit' to stop.\")\n",
        "\n",
        "input_box = widgets.Text(placeholder='Please enter your question:')\n",
        "input_box.on_submit(on_submit)\n",
        "\n",
        "display(input_box)"
      ],
      "metadata": {
        "id": "-pHw5siewPNt"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}