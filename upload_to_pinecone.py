import os
import json
import uuid
from typing import List, Dict
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("nimonik-rag")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > start + chunk_size // 2:
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks

def upload_documents(documents: List[Dict[str, str]]):
    vectors_to_upload = []
    
    for doc in documents:
        chunks = chunk_text(doc['text'])
        
        for i, chunk in enumerate(chunks):
            print(f"Generating embedding for chunk {i+1}/{len(chunks)} of document: {doc.get('title', 'Untitled')}")
            embedding = embedding_model.encode(chunk).tolist()
            
            vector_id = f"{doc.get('id', str(uuid.uuid4()))}_chunk_{i}"
            
            metadata = {
                'text': chunk,
                'title': doc.get('title', 'Untitled'),
                'source': doc.get('source', 'Unknown'),
                'chunk_index': i,
                'total_chunks': len(chunks)
            }
            
            if 'metadata' in doc:
                metadata.update(doc['metadata'])
            
            vectors_to_upload.append({
                'id': vector_id,
                'values': embedding,
                'metadata': metadata
            })
    
    batch_size = 100
    for i in range(0, len(vectors_to_upload), batch_size):
        batch = vectors_to_upload[i:i + batch_size]
        print(f"Uploading batch {i//batch_size + 1}/{(len(vectors_to_upload) + batch_size - 1)//batch_size}")
        
        try:
            index.upsert(vectors=batch)
            print(f"Successfully uploaded {len(batch)} vectors")
        except Exception as e:
            print(f"Error uploading batch: {e}")
    
    print(f"Upload complete! Total vectors uploaded: {len(vectors_to_upload)}")

def upload_from_file(file_path: str, title: str = None, source: str = None):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if not title:
            title = os.path.basename(file_path)
        
        if not source:
            source = file_path
        
        document = {
            'id': str(uuid.uuid4()),
            'title': title,
            'source': source,
            'text': text
        }
        
        upload_documents([document])
        
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

sample_documents = [
    {
        'id': 'legus-food-1',
        'title': 'Legus Favorite Foods',
        'source': 'Personal Preferences',
        'text': '''
        Legus absolutely loves Italian cuisine, especially authentic pasta dishes. His favorite pasta is carbonara with perfectly cooked spaghetti, crispy pancetta, and a rich egg-based sauce. He also enjoys homemade pizza with fresh mozzarella and basil.

        When it comes to desserts, Legus has a sweet tooth for tiramisu and gelato. He particularly enjoys pistachio and stracciatella flavors. For breakfast, he prefers a hearty English breakfast with eggs, bacon, and toast.

        Legus also enjoys experimenting with different cuisines. He recently discovered a love for Japanese ramen and Korean barbecue. His go-to comfort food is his grandmother's chicken soup recipe.
        '''
    },
    {
        'id': 'legus-activities-1',
        'title': 'Legus Favorite Activities',
        'source': 'Personal Interests',
        'text': '''
        Legus is passionate about outdoor activities and adventure sports. His favorite hobby is rock climbing, which he does both indoors and outdoors. He has climbed several challenging routes in the local mountains and dreams of climbing El Capitan in Yosemite.

        When he's not climbing, Legus enjoys hiking and camping. He has completed several multi-day backpacking trips and loves exploring national parks. Photography is another passion of his, especially landscape and wildlife photography during his outdoor adventures.

        Legus also enjoys playing guitar and has been learning for over five years. He particularly likes playing acoustic folk and rock music. He's part of a local music group that meets weekly to jam and perform at small venues.
        '''
    },
    {
        'id': 'legus-tech-1',
        'title': 'Legus Technology Interests',
        'source': 'Professional Profile',
        'text': '''
        Legus is deeply interested in artificial intelligence and machine learning. He has been working on several projects involving natural language processing and computer vision. His current focus is on building RAG (Retrieval-Augmented Generation) systems for document processing.

        He is proficient in Python, JavaScript, and has experience with cloud platforms like AWS and Azure. Legus enjoys building web applications and has created several full-stack projects using React and Node.js. He's particularly interested in the intersection of AI and web development.

        Legus is always learning new technologies and recently started exploring Rust for systems programming. He believes in the importance of clean code and follows best practices in software development. He's also interested in DevOps and has experience with Docker and Kubernetes.
        '''
    },
    {
        'id': 'legus-travel-1',
        'title': 'Legus Travel Experiences',
        'source': 'Personal Stories',
        'text': '''
        Legus loves to travel and has visited over 15 countries. His most memorable trip was to Japan, where he spent three weeks exploring Tokyo, Kyoto, and the Japanese Alps. He was fascinated by the blend of traditional culture and modern technology.

        Another favorite destination is Iceland, where he went on a two-week road trip around the Ring Road. He was amazed by the dramatic landscapes, from glaciers to geysers, and the Northern Lights. He also enjoyed the local cuisine, especially the fresh seafood and lamb dishes.

        Legus prefers immersive travel experiences over touristy attractions. He likes staying in local accommodations, trying authentic food, and learning basic phrases in the local language. His next planned trip is to Patagonia for hiking and wildlife photography.
        '''
    },
    {
        'id': 'legus-goals-1',
        'title': 'Legus Future Goals',
        'source': 'Personal Aspirations',
        'text': '''
        Legus has ambitious goals for his career in technology. He wants to become a senior AI engineer and eventually start his own tech company focused on making AI more accessible to small businesses. He's particularly interested in developing tools that help non-technical users leverage AI capabilities.

        On a personal level, Legus wants to complete a thru-hike of the Pacific Crest Trail, which would take about five months. He's also planning to learn Spanish fluently and hopes to spend time living in South America to immerse himself in the language and culture.

        Legus believes in continuous learning and personal growth. He's committed to reading at least one book per month and attending tech conferences to stay updated with industry trends. He also wants to mentor other developers and contribute to open-source projects.
        '''
    }
]

if __name__ == "__main__":
    print("Uploading sample documents to Pinecone...")
    print(f"Using embedding model: all-MiniLM-L6-v2")
    print(f"Index: nimonik-rag")
    upload_documents(sample_documents)
    print("\nUpload complete!")
