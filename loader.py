from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
)

def load_documents(file_path = "./docs"):
    print("Loading pdf files..")
    loader = PyPDFDirectoryLoader(file_path)
    docs = loader.load()
    return docs

def split_text(chunk_size= 1000, chunk_overlap=200, docs=[]):
    print("Splitting text..")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    return all_splits

def embedding(persist_directory= "./chroma", document=[]):
    print("Embedding..")
    vector_store = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
    ids = vector_store.add_documents(documents=document)




