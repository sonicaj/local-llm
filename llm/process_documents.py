import pathlib
import requests
import typing

import pandas as pd

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from llm.config import PDF_DIR, CSV_DIR, INDEX_DIR, EMBEDDING_MODEL, LLAMA3_BASE_URL


def load_pdf_documents() -> typing.List[Document]:
    pdf_files = [entry.as_posix() for entry in pathlib.Path(PDF_DIR).iterdir() if entry.name.endswith('.pdf')]
    loaders = [PyPDFLoader(pdf) for pdf in pdf_files]
    all_documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    for loader in loaders:
        data = loader.load()
        documents = splitter.split_documents(data)
        all_documents.extend(documents)

    return all_documents


def load_csv_documents() -> typing.List[Document]:
    csv_files = [entry.as_posix() for entry in pathlib.Path(CSV_DIR).iterdir() if entry.name.endswith('.csv')]
    loaders = [CSVLoader(csv) for csv in csv_files]
    all_documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    for loader in loaders:
        data = loader.load()
        documents = splitter.split_documents(data)
        all_documents.extend(documents)

    '''
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df['content'] = df.apply(lambda row: ' '.join(map(str, row)), axis=1)
        for content in df['content']:
            documents = [{'page_content': doc} for doc in splitter.split_text(content)]
            all_documents.extend(documents)
    '''

    return all_documents


def get_remote_embeddings(documents, url):
    """Fetch embeddings from a remote service."""
    responses = []
    for doc in documents:
        response = requests.post(url, json={'text': doc})
        if response.status_code == 200:
            embedding = response.json().get('embedding')
            responses.append(embedding)
        else:
            raise Exception('Failed to get embeddings from the server')
    return responses


def index_documents() -> None:
    """Index both PDF and CSV documents."""
    # Load and combine all document types
    pdf_documents = load_pdf_documents()
    csv_documents = load_csv_documents()
    all_documents = pdf_documents + csv_documents

    # Generate embeddings
    embeddings = get_remote_embeddings(all_documents, f'{LLAMA3_BASE_URL}/api/embeddings')
    # embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Create a FAISS vector store
    db = FAISS.from_documents(all_documents, embeddings)
    db.save_local(INDEX_DIR)
    print(f'Documents indexed and saved to {INDEX_DIR}')
