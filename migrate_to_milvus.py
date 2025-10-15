import os
import json
import pickle
from typing import List, Dict, Any, Optional
from pymilvus import (
    connections, Collection, FieldSchema, CollectionSchema, DataType,
    utility, MilvusException
)
from dotenv import load_dotenv

load_dotenv()

class MilvusManager:
    def __init__(self, host: str = "localhost", port: str = "19530"):
        """
        Initialize Milvus connection and manager.
        
        Args:
            host: Milvus server host
            port: Milvus server port
        """
        self.host = host
        self.port = port
        self.collection_name = "nimonik_rag"
        self.collection = None
        
    def connect(self):
        """Connect to Milvus server."""
        try:
            connections.connect("default", host=self.host, port=self.port)
            print(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            raise
    
    def create_collection_schema(self) -> CollectionSchema:
        """
        Create collection schema matching Pinecone structure.
        
        Returns:
            CollectionSchema object for the RAG collection
        """
        fields = [
            FieldSchema(
                name="id", 
                dtype=DataType.VARCHAR, 
                max_length=512, 
                is_primary=True,
                description="Unique identifier for the vector"
            ),
            FieldSchema(
                name="embedding", 
                dtype=DataType.FLOAT_VECTOR, 
                dim=384,
                description="Text embedding vector"
            ),
            FieldSchema(
                name="text", 
                dtype=DataType.VARCHAR, 
                max_length=65535,
                description="Original text content"
            ),
            FieldSchema(
                name="title", 
                dtype=DataType.VARCHAR, 
                max_length=1024,
                description="Document title"
            ),
            FieldSchema(
                name="source", 
                dtype=DataType.VARCHAR, 
                max_length=1024,
                description="Document source"
            ),
            FieldSchema(
                name="chunk_index", 
                dtype=DataType.INT64,
                description="Chunk index within document"
            ),
            FieldSchema(
                name="total_chunks", 
                dtype=DataType.INT64,
                description="Total number of chunks in document"
            )
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="RAG collection for document embeddings",
            enable_dynamic_field=True
        )
        
        return schema
    
    def create_collection(self, drop_existing: bool = False):
        """
        Create the collection in Milvus.
        
        Args:
            drop_existing: Whether to drop existing collection if it exists
        """
        try:
            if utility.has_collection(self.collection_name):
                if drop_existing:
                    print(f"Dropping existing collection: {self.collection_name}")
                    utility.drop_collection(self.collection_name)
                else:
                    print(f"Collection {self.collection_name} already exists")
                    self.collection = Collection(self.collection_name)
                    return
            
            print(f"Creating collection: {self.collection_name}")
            schema = self.create_collection_schema()
            self.collection = Collection(self.collection_name, schema)
            
            print("Creating index for embedding field...")
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            print(f"Collection {self.collection_name} created successfully")
            
        except Exception as e:
            print(f"Error creating collection: {e}")
            raise
    
    def load_collection(self):
        """Load the collection into memory."""
        if self.collection is None:
            self.collection = Collection(self.collection_name)
        
        self.collection.load()
        print(f"Collection {self.collection_name} loaded into memory")
    
    def insert_data(self, data: List[Dict[str, Any]]):
        """
        Insert data into the collection.
        
        Args:
            data: List of dictionaries containing vector data
        """
        if not self.collection:
            raise ValueError("Collection not initialized")
        
        print(f"Inserting {len(data)} vectors into Milvus...")
        
        ids = []
        embeddings = []
        texts = []
        titles = []
        sources = []
        chunk_indices = []
        total_chunks = []
        
        for item in data:
            ids.append(item['id'])
            embeddings.append(item['values'])
            texts.append(item['metadata'].get('text', ''))
            titles.append(item['metadata'].get('title', ''))
            sources.append(item['metadata'].get('source', ''))
            chunk_indices.append(int(item['metadata'].get('chunk_index', 0)))
            total_chunks.append(int(item['metadata'].get('total_chunks', 1)))
        
        insert_data = [
            ids,
            embeddings,
            texts,
            titles,
            sources,
            chunk_indices,
            total_chunks
        ]
        
        try:
            mr = self.collection.insert(insert_data)
            print(f"Insert completed. Insert count: {mr.insert_count}")
            print(f"Primary keys: {mr.primary_keys[:5]}...")
            
            self.collection.flush()
            print("Data flushed to disk")
            
        except Exception as e:
            print(f"Error inserting data: {e}")
            raise
    
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of search results
        """
        if not self.collection:
            raise ValueError("Collection not initialized")
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text", "title", "source", "chunk_index", "total_chunks"]
        )
        
        formatted_results = []
        for hit in results[0]:
            result = {
                'id': hit.id,
                'score': hit.score,
                'text': hit.entity.get('text'),
                'title': hit.entity.get('title'),
                'source': hit.entity.get('source'),
                'chunk_index': hit.entity.get('chunk_index'),
                'total_chunks': hit.entity.get('total_chunks')
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self.collection:
            raise ValueError("Collection not initialized")
        
        stats = self.collection.num_entities
        return {"total_vectors": stats}
    
    def disconnect(self):
        """Disconnect from Milvus."""
        connections.disconnect("default")
        print("Disconnected from Milvus")

def migrate_from_pinecone(export_file: str = "pinecone_export.json"):
    """
    Migrate data from Pinecone export file to Milvus.
    
    Args:
        export_file: Path to the Pinecone export file
    """
    print("Starting migration from Pinecone to Milvus...")
    print("=" * 60)
    
    print(f"Loading data from {export_file}")
    if export_file.endswith('.pkl'):
        with open(export_file, 'rb') as f:
            data = pickle.load(f)
    else:
        with open(export_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    print(f"Loaded {len(data)} vectors from export file")
    
    milvus_manager = MilvusManager()
    
    try:
        milvus_manager.connect()
        
        milvus_manager.create_collection(drop_existing=True)
        
        milvus_manager.load_collection()
        
        milvus_manager.insert_data(data)
        
        stats = milvus_manager.get_collection_stats()
        print(f"Migration completed! Total vectors in Milvus: {stats['total_vectors']}")
        
        print("\nTesting search functionality...")
        if data:
            test_vector = data[0]['values']
            results = milvus_manager.search(test_vector, top_k=3)
            print(f"Search test returned {len(results)} results")
            for i, result in enumerate(results):
                print(f"Result {i+1}: {result['title']} (score: {result['score']:.4f})")
        
    except Exception as e:
        print(f"Migration failed: {e}")
        raise
    finally:
        milvus_manager.disconnect()

if __name__ == "__main__":
    print("Milvus Migration Tool")
    print("=" * 50)
    
    export_file = "pinecone_export.json"
    if not os.path.exists(export_file):
        print(f"Export file {export_file} not found.")
        print("Please run export_pinecone_data.py first to export data from Pinecone.")
        exit(1)
    
    migrate_from_pinecone(export_file)
    
    print("\nMigration completed successfully!")
    print("You can now use Milvus instead of Pinecone for your RAG system.")
