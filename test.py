import chromadb
import os
import polars as pl
import chromadb.utils.embedding_functions as embedding_functions
from pprint import pprint
from dotenv import load_dotenv

chroma_client = chromadb.PersistentClient(path='./vectordb')

load_dotenv()
API_KEY = os.getenv("OPEN_API_KEY")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=API_KEY,
    model_name="text-embedding-3-small"
)

articles = pl.read_csv("./Articles.csv", encoding='ISO-8859-1').with_row_index(offset=1)
articles.head()
articles = articles[:50]

articles_list = articles['Article'][0:].to_list()
vectors = openai_ef(articles_list)

ids = [f"id{x}" for x in articles['index'][0:].to_list()]

collections = chroma_client.get_or_create_collection(name="articles")

collections.add(
    documents= articles_list,
    ids=ids,
    embeddings= vectors
)


query = 'public transport fares by 7 per cent'
query_embeddings = openai_ef(query)

result = collections.query(
    query_embeddings=query_embeddings,
    n_results = 3,
)

print(result)







