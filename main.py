from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from transformers import pipeline


loader = TextLoader("chatgpt.txt")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
text =text_splitter.split_documents(pages)[:5]








qa_pipeline = pipeline(
    "text2text-generation",
    model= "google/flan-t5-small"
)

question= "What is the life expectancy of United States citizen?"
response = qa_pipeline(question, max_new_token=100)

print(text)