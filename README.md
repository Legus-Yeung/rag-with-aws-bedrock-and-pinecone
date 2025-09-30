# RAG with AWS Bedrock and Pinecone

A Retrieval-Augmented Generation (RAG) system built with AWS Bedrock, Pinecone vector database, and Qwen language models. This project demonstrates how to create a knowledge base and use function calling to enhance AI responses with relevant context.

## 📝 Result

(1) Converting word document about Legus to vector using all-MiniLM-L6-v2 embedding model and uploading to pinecone

<img width="770" height="224" alt="Screenshot 2025-09-30 113651" src="https://github.com/user-attachments/assets/5ee7f68f-b133-4335-b43e-060fcf650699" />

(2) Asking AWS Bedrock hosted Qwen model a question without RAG (as expected, it wouldn't be able to answer without the tool and access to knowledge base)

<img width="1021" height="189" alt="Screenshot 2025-09-30 113728" src="https://github.com/user-attachments/assets/d5de0f60-2a50-477c-bc10-ef60dcaedf5c" />

(3) Asking AWS Bedrock hosted Qwen the same question and it successfully answered with RAG.

<img width="1225" height="585" alt="Screenshot 2025-09-30 114019" src="https://github.com/user-attachments/assets/6c5658a6-b97a-4bcb-b1e2-c997d8c9b879" />

## 🚀 Features

- **Vector Database Integration**: Uses Pinecone for efficient similarity search
- **Function Calling**: Leverages Qwen model's function calling capabilities for intelligent RAG
- **AWS Bedrock Integration**: Utilizes AWS Bedrock for LLM inference
- **Smart Document Chunking**: Intelligent text chunking with overlap for better context
- **Command Line Interface**: Easy-to-use CLI for both simple chat and RAG queries

## 📁 Project Structure

```
rag-with-aws-bedrock-and-pinecone/
├── aws-chat.py          # Simple chat interface with Qwen model
├── aws-chat-rag.py      # RAG system with function calling
├── upload_to_pinecone.py # Document upload and vectorization
├── README.md            # This file
└── venv/                # Python virtual environment
```

## 🛠️ Prerequisites

- Python 3.11+
- AWS Account with Bedrock access
- Pinecone account and API key
- Required Python packages (see installation below)

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Legus-Yeung/rag-with-aws-bedrock-and-pinecone.git
   cd rag-with-aws-bedrock-and-pinecone
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install boto3 python-dotenv pinecone-client sentence-transformers
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   PINECONE_API_KEY=your_pinecone_api_key_here
   AWS_ACCESS_KEY_ID=your_aws_access_key
   AWS_SECRET_ACCESS_KEY=your_aws_secret_key
   AWS_DEFAULT_REGION=us-east-1
   ```

## 🚀 Quick Start

### 1. Upload Documents to Pinecone

First, upload sample documents to create your knowledge base:

```bash
python upload_to_pinecone.py
```

This will upload sample documents about "Legus" to your Pinecone index named "nimonik-rag".

### 2. Simple Chat (No RAG)

For basic chat without retrieval:

```bash
python aws-chat.py "What is artificial intelligence?"
```

### 3. RAG with Function Calling

For intelligent retrieval-augmented responses:

```bash
python aws-chat-rag.py "What are Legus's favorite foods?"
```

The system will:
1. Check if the AI knows the answer
2. If not, automatically call the search function
3. Retrieve relevant documents from Pinecone
4. Provide an enhanced answer with context

## 🔧 Configuration

### Model Configuration

The system uses the Qwen 3 Coder 30B model on AWS Bedrock:
- **Model ID**: `qwen.qwen3-coder-30b-a3b-v1:0`
- **Region**: `us-east-1`
- **Max Tokens**: 512

### Vector Database Configuration

- **Embedding Model**: `all-MiniLM-L6-v2`
- **Chunk Size**: 1000 characters
- **Overlap**: 200 characters
- **Index Name**: `nimonik-rag`

## 📚 How It Works

### RAG System Flow

1. **User Query**: User asks a question via command line
2. **Initial Assessment**: AI evaluates if it knows the answer
3. **Function Calling**: If needed, AI calls `search_knowledge_base` function
4. **Vector Search**: System searches Pinecone for relevant documents
5. **Context Integration**: Retrieved context is sent back to AI
6. **Final Response**: AI provides answer with retrieved context

### Document Processing

1. **Text Chunking**: Documents are split into overlapping chunks
2. **Embedding Generation**: Each chunk is converted to vector embeddings
3. **Vector Storage**: Embeddings stored in Pinecone with metadata
4. **Similarity Search**: Queries are embedded and matched against stored vectors

## 🎯 Function Calling Implementation

The system implements function calling with the following structure:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the knowledge base for relevant information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 3}
                },
                "required": ["query"]
            }
        }
    }
]
```

## 📝 Adding Your Own Documents

To add your own documents, modify the `sample_documents` list in `upload_to_pinecone.py`:

```python
custom_documents = [
    {
        'id': 'unique-id',
        'title': 'Document Title',
        'source': 'Source Information',
        'text': 'Your document content here...',
        'metadata': {
            'category': 'optional',
            'tags': ['tag1', 'tag2']
        }
    }
]
```

Or upload from a file:

```python
upload_from_file('path/to/your/document.txt', title='Custom Title')
```

## 🔍 Example Queries

Try these example queries to test the system. **For best comparison, test the same questions with both simple chat and RAG versions** to see the difference:

```bash
# Test the same questions with both versions:

# Simple chat (no RAG)
python aws-chat.py "What are Legus's favorite outdoor activities?"
python aws-chat.py "What programming languages does Legus know?"
python aws-chat.py "Where has Legus traveled?"
python aws-chat.py "What are Legus's career goals?"

# RAG queries (with function calling)
python aws-chat-rag.py "What are Legus's favorite outdoor activities?"
python aws-chat-rag.py "What programming languages does Legus know?"
python aws-chat-rag.py "Where has Legus traveled?"
python aws-chat-rag.py "What are Legus's career goals?"

# General knowledge questions (both should work similarly)
python aws-chat.py "Explain machine learning"
python aws-chat-rag.py "Explain machine learning"
```

**Note**: The RAG version will provide more detailed, context-specific answers for questions about Legus, while the simple chat version will give general responses or say "I don't know" for specific personal information.

## 🐛 Troubleshooting

### Common Issues

1. **AWS Credentials Error**
   - Ensure AWS credentials are properly configured
   - Check that Bedrock is available in your region

2. **Pinecone Connection Error**
   - Verify your Pinecone API key
   - Ensure the index "nimonik-rag" exists

3. **Model Not Found**
   - Check if Qwen model is available in your AWS region
   - Verify model ID is correct

4. **Import Errors**
   - Ensure all dependencies are installed
   - Activate the virtual environment

### Debug Mode

Add debug prints to see what's happening:

```python
print(f"Response: {response}")
print(f"Tool calls: {response.get('tool_calls', [])}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [AWS Bedrock](https://aws.amazon.com/bedrock/) for LLM hosting
- [Pinecone](https://www.pinecone.io/) for vector database
- [Qwen](https://qwen.readthedocs.io/) for the language model
- [Sentence Transformers](https://www.sbert.net/) for embeddings

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review AWS Bedrock and Pinecone documentation
3. Open an issue in this repository

---

**Happy RAG-ing! 🎉**
