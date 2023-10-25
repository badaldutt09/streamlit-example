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
