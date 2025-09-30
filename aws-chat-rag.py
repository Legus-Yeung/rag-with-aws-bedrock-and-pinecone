import boto3
import json
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

client = boto3.client("bedrock-runtime", region_name="us-east-1")
model_id = "qwen.qwen3-coder-30b-a3b-v1:0"

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("nimonik-rag")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    """Generate embedding for the given text"""
    return embedding_model.encode(text).tolist()

def retrieve_relevant_docs(query, top_k=3):
    """Retrieve relevant documents from Pinecone based on the query"""
    query_embedding = get_embedding(query)

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    relevant_docs = []
    for match in results.matches:
        relevant_docs.append({
            'content': match.metadata.get('text', ''),
            'score': match.score
        })
    
    return relevant_docs

def create_rag_prompt(user_query, relevant_docs):
    """Create a prompt that includes retrieved context"""
    context = "\n\n".join([doc['content'] for doc in relevant_docs])
    
    prompt = f"""Based on the following context, please answer the user's question:

Context:
{context}

User Question: {user_query}

Please provide a helpful answer based on the context provided. If the context doesn't contain relevant information, you can use your general knowledge but mention that the context didn't contain specific information about this topic."""
    
    return prompt

user_prompt = "What are Legus's favorite foods and cuisines?"

print("Retrieving relevant documents...")
relevant_docs = retrieve_relevant_docs(user_prompt, top_k=3)

print(f"Found {len(relevant_docs)} relevant documents")
for i, doc in enumerate(relevant_docs):
    print(f"Document {i+1} (score: {doc['score']:.3f}): {doc['content'][:100]}...")

rag_prompt = create_rag_prompt(user_prompt, relevant_docs)

print("\nSending request to Bedrock...")
body = {
    "messages": [
        {"role": "user", "content": rag_prompt}
    ],
    "max_tokens": 512
}

response = client.invoke_model(
    modelId=model_id,
    body=json.dumps(body)
)

response_body = json.loads(response["body"].read())

if "choices" in response_body and len(response_body["choices"]) > 0:
    assistant_content = response_body["choices"][0]["message"]["content"]
    print("\nAssistant:", assistant_content)
else:
    print("Error: No choices found in response")
    print("Available keys:", list(response_body.keys()))
