import chromadb
from sentence_transformers import SentenceTransformer
import os
import glob

class RAGModule:
    def __init__(self, collection_name="knowledge_base", persistence_path="./chroma_db"):
        print(f"Initializing RAG Module ({collection_name})...")
        self.client = chromadb.PersistentClient(path=persistence_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        # Use a lightweight model for local embedding
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"RAG Module ({collection_name}) initialized.")

    def ingest(self, directory_path):
        """Reads all .txt files in the directory and adds them to the vector store."""
        files = glob.glob(os.path.join(directory_path, "*.txt"))
        print(f"Found {len(files)} documents to ingest.")
        
        documents = []
        metadatas = []
        ids = []

        for i, file_path in enumerate(files):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append(content)
                metadatas.append({"source": os.path.basename(file_path)})
                ids.append(f"doc_{i}")
        
        if documents:
            # Generate embeddings
            embeddings = self.embedder.encode(documents).tolist()
            
            # Upsert into Chroma
            self.collection.upsert(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Ingested {len(documents)} documents into ChromaDB.")
        else:
            print("No documents found to ingest.")

    def retrieve(self, query, n_results=3):
        """Retrieves the most relevant document segments for the query."""
        query_embedding = self.embedder.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        if results['documents'] and results['documents'][0]:
            # Return lists of (content, metadata)
            return results['documents'][0], results['metadatas'][0]
        return [], []
