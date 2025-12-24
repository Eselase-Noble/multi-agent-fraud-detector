import chromadb
from chromadb.config import Settings

# Create a Chroma client in "in-process" mode
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",   # stores data on disk
    persist_directory="./chroma_db"    # folder to save data
))

# Example: create a collection
collection = client.create_collection("my_collection")

# Add a sample vector
collection.add(
    documents=["Hello world!"],
    embeddings=[[0.1, 0.2, 0.3]],  # your embedding vector
    ids=["doc1"]
)

# Query example
results = collection.query(
    query_embeddings=[[0.1, 0.2, 0.3]],
    n_results=1
)

print(results)
