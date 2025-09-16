import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable, Dict, Any, Optional
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio


def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


def euclidean_distance(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the euclidean distance between two vectors."""
    return np.linalg.norm(vector_a - vector_b)


def manhattan_distance(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the manhattan distance between two vectors."""
    return np.sum(np.abs(vector_a - vector_b))


class VectorWithMetadata:
    """Container for vector with associated metadata"""
    def __init__(self, vector: np.array, metadata: Dict[str, Any]):
        self.vector = vector
        self.metadata = metadata


class VectorDatabase:
    def __init__(self, embedding_model: EmbeddingModel = None):
        self.vectors = defaultdict(VectorWithMetadata)
        self.embedding_model = embedding_model or EmbeddingModel()
        self.text_to_key = {}  # Maps text content to vector keys
        self.chunk_counter = 0
        
        # Distance functions dictionary
        self.distance_functions = {
            'cosine': cosine_similarity,
            'euclidean': euclidean_distance,
            'manhattan': manhattan_distance
        }

    def insert(self, key: str, vector: np.array, metadata: Dict[str, Any] = None) -> None:
        """Insert a vector with optional metadata"""
        if metadata is None:
            metadata = {}
        self.vectors[key] = VectorWithMetadata(vector, metadata)

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure = None,
        file_type_filter: str = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search vectors with optional file type filtering"""
        # Handle distance measure - can be a function or string name
        if distance_measure is None:
            distance_func = cosine_similarity
        elif isinstance(distance_measure, str):
            distance_func = self.distance_functions.get(distance_measure, cosine_similarity)
        else:
            distance_func = distance_measure
            
        scores = []
        for key, vector_with_metadata in self.vectors.items():
            # Apply file type filter if specified
            if file_type_filter and vector_with_metadata.metadata.get('file_type') != file_type_filter:
                continue
                
            score = distance_func(query_vector, vector_with_metadata.vector)
            scores.append((key, score, vector_with_metadata.metadata))
        
        # For distance metrics (euclidean, manhattan), lower is better, so reverse=False
        # For similarity metrics (cosine), higher is better, so reverse=True
        reverse = distance_measure in ['cosine', None] or (isinstance(distance_measure, str) and distance_measure == 'cosine')
        return sorted(scores, key=lambda x: x[1], reverse=reverse)[:k]

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure = None,
        return_as_text: bool = False,
        include_metadata: bool = False,
        file_type_filter: str = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search by text with optional metadata and file type filtering"""
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure, file_type_filter)
        
        # Convert vector keys back to text content
        text_results = []
        for key, score, metadata in results:
            # Find the text content for this key
            text_content = None
            for text, stored_key in self.text_to_key.items():
                if stored_key == key:
                    text_content = text
                    break
            
            if text_content is None:
                # Fallback: use key as text if no mapping found
                text_content = key
                
            text_results.append((text_content, score, metadata))
        
        if return_as_text and not include_metadata:
            return [result[0] for result in text_results]
        elif return_as_text and include_metadata:
            return [(result[0], result[1], result[2]) for result in text_results]
        else:
            return text_results

    def retrieve_from_key(self, key: str) -> np.array:
        vector_with_metadata = self.vectors.get(key, None)
        return vector_with_metadata.vector if vector_with_metadata else None

    async def abuild_from_list(self, list_of_text: List[str], metadata_list: List[Dict[str, Any]] = None) -> "VectorDatabase":
        """Build vector database from list of texts with optional metadata"""
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        
        for i, (text, embedding) in enumerate(zip(list_of_text, embeddings)):
            # Create unique key for this chunk
            chunk_key = f"chunk_{self.chunk_counter}"
            self.chunk_counter += 1
            
            # Get metadata for this chunk
            chunk_metadata = {}
            if metadata_list and i < len(metadata_list):
                # Convert DocumentMetadata to dict if needed
                if hasattr(metadata_list[i], 'to_dict'):
                    chunk_metadata = metadata_list[i].to_dict()
                else:
                    chunk_metadata = metadata_list[i]
                chunk_metadata['chunk_index'] = i
            
            # Store text to key mapping
            self.text_to_key[text] = chunk_key
            
            # Insert vector with metadata
            self.insert(chunk_key, np.array(embedding), chunk_metadata)
        
        return self
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        file_types = defaultdict(int)
        total_vectors = len(self.vectors)
        
        for vector_with_metadata in self.vectors.values():
            file_type = vector_with_metadata.metadata.get('file_type', 'unknown')
            file_types[file_type] += 1
        
        return {
            'total_vectors': total_vectors,
            'file_types': dict(file_types),
            'available_distance_metrics': ['cosine', 'euclidean', 'manhattan']
        }


if __name__ == "__main__":
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]

    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(list_of_text))
    k = 2

    searched_vector = vector_db.search_by_text("I think fruit is awesome!", k=k)
    print(f"Closest {k} vector(s):", searched_vector)

    retrieved_vector = vector_db.retrieve_from_key(
        "I like to eat broccoli and bananas."
    )
    print("Retrieved vector:", retrieved_vector)

    relevant_texts = vector_db.search_by_text(
        "I think fruit is awesome!", k=k, return_as_text=True
    )
    print(f"Closest {k} text(s):", relevant_texts)
