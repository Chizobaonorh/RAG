import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from transformers import pipeline

load_dotenv()

openai_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=openai_ef
)

qa_pipeline = pipeline(
    "text2text-generation",
    model= "google/flan-t5-small"
)

collections.add(
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ],
    ids= ["id1", "id2"]
)

results = collections.query(
    query_texts=["This is a query document about Hawaii"],
    n_results=2,
    where_document={'$contains': 'pineapple'}
)

pprint(results)


question= "What is the life expectancy of United States citizen?"
response = qa_pipeline(question, max_new_token=100)

print(response)