import boto3
import json
import os
import sys
from dotenv import load_dotenv
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

load_dotenv()

client = boto3.client("bedrock-runtime", region_name="us-east-1")
model_id = "qwen.qwen3-coder-30b-a3b-v1:0"

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "nimonik_rag"

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class MilvusRAG:
    def __init__(self):
        self.collection = None
        self.connected = False
    
    def connect(self):
        """Connect to Milvus and load collection."""
        try:
            connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
            self.collection = Collection(COLLECTION_NAME)
            self.collection.load()
            self.connected = True
            print(f"Connected to Milvus collection: {COLLECTION_NAME}")
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from Milvus."""
        if self.connected:
            connections.disconnect("default")
            self.connected = False
            print("Disconnected from Milvus")
    
    def get_embedding(self, text):
        """Generate embedding for the given text"""
        return embedding_model.encode(text).tolist()
    
    def retrieve_relevant_docs(self, query, top_k=3):
        """Retrieve relevant documents from Milvus based on the query"""
        if not self.connected:
            raise ValueError("Not connected to Milvus")
        
        query_embedding = self.get_embedding(query)
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text", "title", "source", "chunk_index", "total_chunks"]
        )
        
        relevant_docs = []
        for hit in results[0]:
            relevant_docs.append({
                'content': hit.entity.get('text', ''),
                'title': hit.entity.get('title', ''),
                'source': hit.entity.get('source', ''),
                'chunk_index': hit.entity.get('chunk_index', 0),
                'total_chunks': hit.entity.get('total_chunks', 1),
                'score': hit.score
            })
        
        return relevant_docs

milvus_rag = MilvusRAG()

def create_rag_prompt(user_query, relevant_docs):
    """Create a prompt that includes retrieved context"""
    context = "\n\n".join([doc['content'] for doc in relevant_docs])
    
    prompt = f"""Based on the following context, please answer the user's question:

Context:
{context}

User Question: {user_query}

Please provide a helpful answer based on the context provided. If the context doesn't contain relevant information, you can use your general knowledge but mention that the context didn't contain specific information about this topic."""
    
    return prompt

def ask_ai(prompt):
    """Send a prompt to the AI and return the response"""
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    body = {
        "messages": messages,
        "max_tokens": 512
    }

    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body)
    )

    response_body = json.loads(response["body"].read())

    if "choices" in response_body and len(response_body["choices"]) > 0:
        return response_body["choices"][0]["message"]["content"]
    else:
        return "Error: No response from AI"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python aws-chat-rag-milvus-simple.py <your_question>")
        print("Example: python aws-chat-rag-milvus-simple.py 'what are Legus favorite foods'")
        sys.exit(1)

    user_prompt = " ".join(sys.argv[1:])

    print(f"Question: {user_prompt}")
    print("\nConnecting to Milvus...")
    
    try:
        milvus_rag.connect()
        
        print("Searching knowledge base...")
        
        relevant_docs = milvus_rag.retrieve_relevant_docs(user_prompt, top_k=3)
        
        print(f"Found {len(relevant_docs)} relevant documents")
        for i, doc in enumerate(relevant_docs):
            print(f"Document {i+1} (score: {doc['score']:.3f}): {doc['title']}")
        
        if relevant_docs:
            rag_prompt = create_rag_prompt(user_prompt, relevant_docs)
            
            print("\nGenerating response with context...")
            response = ask_ai(rag_prompt)
            print(f"\nAssistant Response: {response}")
        else:
            print("No relevant documents found. Asking AI without context...")
            response = ask_ai(user_prompt)
            print(f"\nAssistant Response: {response}")
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        milvus_rag.disconnect()
