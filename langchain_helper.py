from langchain.document_loaders import PyPDFLoader
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from dotenv import load_dotenv
import os


load_dotenv()

file_path = './REBGV-Dec-2023.pdf'

local_persist_path = './vector_store'

def get_index_path(index_name):
    return os.path.join(local_persist_path, index_name)

def load_pdf_and_save_to_index(file_path, index_name):
    loader = PyPDFLoader(file_path)
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":get_index_path(index_name)}).from_loaders([loader])
    index.vectorstore.persist()
    
# load_pdf_and_save_to_index(file_path, 'REBGV-Dec-2023.pdf')

def load_index(index_name):
    index_path = get_index_path(index_name)
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(
        persist_directory=index_path,
        embedding_function=embeddings,
    )
    return VectorStoreIndexWrapper(vectorstore=vectordb)

index = load_index('REBGV-Dec-2023.pdf')

ans = index.query_with_sources("what is the trend of the vancouver realestate market DEC 2023?",chain_type="map_reduce")

print(ans)