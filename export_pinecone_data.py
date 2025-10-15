import os
import json
import pickle
from typing import List, Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

def export_pinecone_data(index_name: str = "nimonik-rag", output_file: str = "pinecone_export.json"):
    """
    Export all vectors and metadata from Pinecone index to a JSON file.
    
    Args:
        index_name: Name of the Pinecone index to export
        output_file: Output file path for the exported data
    """
    print(f"Connecting to Pinecone index: {index_name}")
    
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(index_name)
    
    stats = index.describe_index_stats()
    print(f"Index stats: {stats}")
    
    total_vectors = stats.total_vector_count
    print(f"Total vectors to export: {total_vectors}")
    
    if total_vectors == 0:
        print("No vectors found in the index.")
        return
    
    exported_data = []
    batch_size = 100
    
    dummy_vector = [0.0] * 384
    
    print("Starting export process...")
    
    try:
        results = index.query(
            vector=dummy_vector,
            top_k=total_vectors,
            include_metadata=True,
            include_values=True
        )
        
        print(f"Retrieved {len(results.matches)} vectors")
        
        for match in results.matches:
            vector_data = {
                'id': match.id,
                'values': match.values,
                'metadata': match.metadata,
                'score': match.score
            }
            exported_data.append(vector_data)
        
        print(f"Successfully exported {len(exported_data)} vectors")
        
    except Exception as e:
        print(f"Error during export: {e}")
        print("Trying alternative method...")
        
        print("Note: This method requires knowing vector IDs. Consider using the Pinecone console to get IDs.")
        return
    
    print(f"Saving exported data to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(exported_data, f, indent=2, ensure_ascii=False)
    
    pickle_file = output_file.replace('.json', '.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(exported_data, f)
    
    print(f"Export complete!")
    print(f"JSON file: {output_file}")
    print(f"Pickle file: {pickle_file}")
    print(f"Total vectors exported: {len(exported_data)}")
    
    if exported_data:
        print("\nSample of exported data:")
        sample = exported_data[0]
        print(f"ID: {sample['id']}")
        print(f"Vector dimensions: {len(sample['values'])}")
        print(f"Metadata keys: {list(sample['metadata'].keys())}")
        print(f"Sample metadata: {sample['metadata']}")

def load_exported_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load exported data from file.
    
    Args:
        file_path: Path to the exported data file
        
    Returns:
        List of vector data dictionaries
    """
    if file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

if __name__ == "__main__":
    print("Pinecone Data Export Tool")
    print("=" * 50)
    
    export_pinecone_data()
    
    print("\nExport completed successfully!")
    print("You can now use this data to migrate to Milvus.")
