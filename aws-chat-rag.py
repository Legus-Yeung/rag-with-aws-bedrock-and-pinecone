import boto3
import json
import os
import sys
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

def create_initial_prompt(user_query):
    """Create initial prompt to check if AI knows the answer"""
    prompt = f"""Please answer the following question: {user_query}

If you don't have enough information to provide a confident answer, please respond with exactly "I don't know" and nothing else. Otherwise, provide your answer based on your knowledge."""
    
    return prompt

def create_system_prompt_with_tools():
    """Create system prompt that instructs AI to use function calling when needed"""
    return """You are a helpful assistant with access to a knowledge base search function. 

When a user asks a question:
1. First, try to answer based on your general knowledge
2. If you don't know the answer or need more specific information, use the search_knowledge_base function to find relevant information
3. Then provide a comprehensive answer based on the search results

Always be helpful and provide accurate information."""

def create_rag_prompt(user_query, relevant_docs):
    """Create a prompt that includes retrieved context"""
    context = "\n\n".join([doc['content'] for doc in relevant_docs])
    
    prompt = f"""Based on the following context, please answer the user's question:

Context:
{context}

User Question: {user_query}

Please provide a helpful answer based on the context provided. If the context doesn't contain relevant information, you can use your general knowledge but mention that the context didn't contain specific information about this topic."""
    
    return prompt

def create_search_tool():
    """Create the search tool definition for function calling"""
    return [
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": "Search the knowledge base for relevant information about a given query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant information"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of top results to return (default: 3)",
                            "default": 3
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]

def handle_function_call(function_name, arguments):
    """Handle function calls from the AI"""
    if function_name == "search_knowledge_base":
        query = arguments.get("query", "")
        top_k = arguments.get("top_k", 3)
        
        print(f"\nSearching knowledge base for: {query}")
        relevant_docs = retrieve_relevant_docs(query, top_k=top_k)
        
        print(f"Found {len(relevant_docs)} relevant documents")
        for i, doc in enumerate(relevant_docs):
            print(f"Document {i+1} (score: {doc['score']:.3f}): {doc['content'][:100]}...")
        
        context = "\n\n".join([doc['content'] for doc in relevant_docs])
        return f"Search results for '{query}':\n\n{context}"
    
    return "Function not found"

def ask_ai(prompt, tools=None, system_prompt=None):
    """Send a prompt to the AI and return the response"""
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    body = {
        "messages": messages,
        "max_tokens": 512
    }
    
    if tools:
        body["tools"] = tools

    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body)
    )

    response_body = json.loads(response["body"].read())

    if "choices" in response_body and len(response_body["choices"]) > 0:
        return response_body["choices"][0]["message"]
    else:
        return {"content": "Error: No response from AI"}

if len(sys.argv) < 2:
    print("Usage: python aws-chat-rag.py <your_question>")
    print("Example: python aws-chat-rag.py 'what is the temperature in amsterdam'")
    sys.exit(1)

user_prompt = " ".join(sys.argv[1:])

print(f"Question: {user_prompt}")
print("\nUsing function calling to answer the question...")

system_prompt = create_system_prompt_with_tools()
tools = create_search_tool()

response = ask_ai(user_prompt, tools=tools, system_prompt=system_prompt)

if "tool_calls" in response and response["tool_calls"]:
    print("\nAI is calling functions to search for information...")
    
    for tool_call in response["tool_calls"]:
        function_name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])
        
        function_result = handle_function_call(function_name, arguments)
        
        follow_up_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            response,
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": function_result
            }
        ]
        
        body = {
            "messages": follow_up_messages,
            "max_tokens": 512
        }
        
        final_response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(body)
        )
        
        final_response_body = json.loads(final_response["body"].read())
        
        if "choices" in final_response_body and len(final_response_body["choices"]) > 0:
            final_content = final_response_body["choices"][0]["message"]["content"]
            print("\nFinal Assistant Response:", final_content)
        else:
            print("\nError: No final response from AI")
else:
    print("\nAI provided an answer without needing to search.")
    print("Final Assistant Response:", response.get("content", "No content in response"))
