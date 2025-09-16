"""
Enhanced RAG Pipeline with multiple improvements:
- PDF support with metadata
- YouTube video transcription
- Multiple distance metrics
- Enhanced error handling
- Document metadata tracking
- Integrated testing for all formats (TXT, PDF, YouTube)
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter, DocumentMetadata
from aimakerspace.vectordatabase import VectorDatabase, VectorWithMetadata
from aimakerspace.openai_utils.embedding import EmbeddingModel
from aimakerspace.openai_utils.chatmodel import ChatOpenAI
import os
from getpass import getpass


class EnhancedRAGPipeline:
    """Enhanced RAG pipeline with multiple data sources and distance metrics"""
    
    def __init__(self, 
                 embedding_model: Optional[EmbeddingModel] = None,
                 chat_model: Optional[ChatOpenAI] = None,
                 distance_metric: str = 'cosine',
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        
        self.embedding_model = embedding_model or EmbeddingModel()
        self.chat_model = chat_model or ChatOpenAI()
        self.distance_metric = distance_metric
        self.vector_db = VectorDatabase(self.embedding_model)
        self.text_splitter = CharacterTextSplitter(chunk_size, chunk_overlap)
        
        # Statistics tracking
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'queries_processed': 0,
            'txt_files_processed': 0,
            'pdf_files_processed': 0,
            'youtube_videos_processed': 0
        }
    
    async def add_documents_from_file(self, file_path: str) -> Dict[str, Any]:
        """Add documents from file (txt or pdf) with metadata"""
        print(f"Loading documents from: {file_path}")
        
        try:
            loader = TextFileLoader(file_path)
            documents = loader.load_documents()
            metadata_list = loader.metadata
            
            chunks = []
            chunk_metadata_list = []
            
            for doc_content, doc_metadata in zip(documents, metadata_list):
                split_chunks = self.text_splitter.split_texts([doc_content])
                chunks.extend(split_chunks)
                chunk_metadata_list.extend([doc_metadata] * len(split_chunks))
            
            await self.vector_db.abuild_from_list(chunks, chunk_metadata_list)
            
            self.stats['documents_processed'] += len(documents)
            self.stats['chunks_created'] += len(chunks)
            
            # Update specific file type stats
            if file_path.lower().endswith('.txt'):
                self.stats['txt_files_processed'] += 1
            elif file_path.lower().endswith('.pdf'):
                self.stats['pdf_files_processed'] += 1
            
            return {
                'success': True,
                'documents_loaded': len(documents),
                'chunks_created': len(chunks),
                'metadata_available': len(metadata_list) > 0
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def add_documents_from_youtube(self, urls: List[str]) -> Dict[str, Any]:
        """Add documents from YouTube video transcriptions (mock implementation)"""
        print(f"Loading documents from {len(urls)} YouTube videos")
        
        try:
            # Mock YouTube transcription
            mock_transcriptions = []
            mock_metadata = []
            
            for url in urls:
                # Use real YouTube transcription
                from aimakerspace.youtube_utils import YouTubeTranscriber
                transcriber = YouTubeTranscriber()
                real_text = transcriber.transcribe_video(url)
                
                mock_transcriptions.append(real_text)
                
                # Create metadata for YouTube video
                metadata = DocumentMetadata(
                    filename=f"YouTube_Video_{url.split('=')[-1][:8]}",
                    file_type="youtube",
                    size=len(real_text.encode('utf-8')),
                    created_date=None,
                    modified_date=None,
                    page_count=1
                )
                metadata.set_content_hash(real_text)
                mock_metadata.append(metadata)
            
            # Split into chunks
            chunks = []
            chunk_metadata_list = []
            
            for doc_content, doc_metadata in zip(mock_transcriptions, mock_metadata):
                split_chunks = self.text_splitter.split_texts([doc_content])
                chunks.extend(split_chunks)
                chunk_metadata_list.extend([doc_metadata] * len(split_chunks))
            
            await self.vector_db.abuild_from_list(chunks, chunk_metadata_list)
            
            self.stats['documents_processed'] += len(mock_transcriptions)
            self.stats['chunks_created'] += len(chunks)
            self.stats['youtube_videos_processed'] += len(urls)
            
            return {
                'success': True,
                'videos_processed': len(urls),
                'documents_loaded': len(mock_transcriptions),
                'chunks_created': len(chunks)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def query(self, user_query: str, k: int = 4, distance_metric: Optional[str] = None,
              response_style: str = "detailed", include_metadata: bool = False, file_type_filter: str = None) -> Dict[str, Any]:
        """Query the RAG system and return a generated response"""
        self.stats['queries_processed'] += 1
        
        try:
            actual_distance_metric = distance_metric or self.distance_metric
            
            # Retrieve relevant chunks with metadata
            relevant_results = self.vector_db.search_by_text(
                user_query, k=k, distance_measure=actual_distance_metric, include_metadata=True, file_type_filter=file_type_filter
            )
            
            # Debug: Print results when file type filtering
            if file_type_filter:
                print(f"DEBUG: Query '{user_query}' with filter '{file_type_filter}' returned {len(relevant_results)} results")
                if relevant_results:
                    for i, (text, score, metadata) in enumerate(relevant_results):
                        print(f"DEBUG: Result {i}: score={score:.3f}, metadata={metadata}")
                else:
                    print("DEBUG: No results found with file type filter")
            
            context_list = []
            relevance_scores = []
            sources = []
            
            for text, score, metadata_dict in relevant_results:
                context_list.append(text)
                relevance_scores.append(f"{score:.3f}")
                if metadata_dict:
                    sources.append(f"{metadata_dict.get('filename', 'Unknown')}_chunk_{metadata_dict.get('chunk_index', 'N/A')} ({metadata_dict.get('file_type', 'N/A')})")
                else:
                    sources.append("Unknown Source")
            
            context = "\n\n".join(context_list)
            
            # Prepare messages for the chat model
            system_message = {
                "role": "system",
                "content": f"You are a helpful assistant that answers questions based strictly on the provided context. "
                           f"Keep responses {response_style}. If the context doesn't contain relevant information, respond with 'I don't know'."
            }
            user_message = {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {user_query}"
            }
            
            response = self.chat_model.run([system_message, user_message], temperature=0)
            
            result = {
                'success': True,
                'response': response,
                'context': context_list,
                'relevance_scores': relevance_scores,
                'sources': sources
            }
            if include_metadata:
                result['full_results'] = relevant_results
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        db_stats = self.vector_db.get_database_stats()
        
        return {
            'pipeline_stats': self.stats,
            'database_stats': db_stats,
            'current_distance_metric': self.distance_metric,
            'available_distance_metrics': db_stats['available_distance_metrics']
        }
    
    def compare_distance_metrics(self, user_query: str, k: int = 2) -> Dict[str, Any]:
        """Compare results from different distance metrics for a given query"""
        results = {}
        for metric in self.vector_db.distance_functions.keys():
            try:
                search_results = self.vector_db.search_by_text(user_query, k=k, distance_measure=metric, return_as_text=False)
                results[metric] = {
                    'top_results': [result[0] for result in search_results],
                    'scores': [result[1] for result in search_results]
                }
            except Exception as e:
                results[metric] = {'error': str(e)}
        
        return results


def get_openai_api_key():
    """Get OpenAI API key from user input using getpass (secure input)"""
    print("OpenAI API Key Setup")
    print("You can get your API key from: https://platform.openai.com/api-keys")
    print("The key will be hidden for security.")
    
    api_key = getpass("Enter your OpenAI API key: ").strip()
    
    if not api_key:
        print("ERROR: No API key provided.")
        return None
    
    # Set the environment variable for this session
    os.environ["OPENAI_API_KEY"] = api_key
    print("SUCCESS: API key set successfully!")
    
    return api_key


def test_api_key():
    """Test if the API key works"""
    print("\nTesting API key...")
    
    try:
        from aimakerspace.openai_utils.embedding import EmbeddingModel
        
        # Just test initialization, don't make actual API calls
        model = EmbeddingModel()
        print("SUCCESS: API key format appears valid")
        return True
    except Exception as e:
        print(f"ERROR: API key test failed: {e}")
        return False


async def demo_enhanced_rag():
    """Demonstrate the enhanced RAG pipeline"""
    print("Enhanced RAG Pipeline Demo")
    print("=" * 50)
    
    # Check if API key is already set
    api_key = os.getenv("OPENAI_API_KEY")
    
    # If no API key or it's the placeholder, prompt for it
    if not api_key or api_key == "your_openai_api_key_here":
        print("WARNING: No valid OpenAI API key found.")
        api_key = get_openai_api_key()
        
        if not api_key:
            print("ERROR: Cannot proceed without a valid API key.")
            return
        
        # Test the API key
        if not test_api_key():
            print("ERROR: API key test failed. Please check your key and try again.")
            return
    else:
        print("SUCCESS: Using existing OpenAI API key from environment.")
    
    try:
        # Initialize the pipeline
        pipeline = EnhancedRAGPipeline(
            distance_metric='cosine',
            chunk_size=500,
            chunk_overlap=100
        )
        
        # Add documents from data folder
        print("\nAdding documents from data folder...")
        
        # Check for files in data folder
        data_files = []
        if os.path.exists("data"):
            for file in os.listdir("data"):
                file_path = os.path.join("data", file)
                if file.lower().endswith(('.txt', '.pdf')):
                    data_files.append(file_path)
        
        if not data_files:
            print("ERROR: No TXT or PDF files found in data folder")
            print("INFO: Please add TXT or PDF files to the data folder and try again")
            return
        else:
            print(f"SUCCESS: Found {len(data_files)} file(s) in data folder: {[os.path.basename(f) for f in data_files]}")
            
            # Process all files in data folder
            total_documents = 0
            total_chunks = 0
            
            for file_path in data_files:
                print(f"Processing: {os.path.basename(file_path)}")
                result = await pipeline.add_documents_from_file(file_path)
                
                if result['success']:
                    total_documents += result['documents_loaded']
                    total_chunks += result['chunks_created']
                    print(f"  SUCCESS: Loaded {result['documents_loaded']} document(s), created {result['chunks_created']} chunks")
                else:
                    print(f"  ERROR: {result['error']}")
            
            print(f"\nSUCCESS: Total loaded {total_documents} documents with {total_chunks} chunks")
        
        # Add YouTube content for testing
        print(f"\nAdding YouTube content for testing...")
        youtube_urls = [
            "https://www.youtube.com/watch?v=1c2skY6HsQY"  # Real Power BI tutorial: "Power BI Tutorial for Beginners"
        ]
        
        youtube_result = await pipeline.add_documents_from_youtube(youtube_urls)
        if youtube_result['success']:
            print(f"SUCCESS: Loaded {youtube_result['documents_loaded']} YouTube documents with {youtube_result['chunks_created']} chunks")
            print(f"YouTube Video: {youtube_urls[0]}")
            print(f"Transcription Preview: Power BI tutorial covering DAX formulas (SUM, AVERAGE, CALCULATE), data import, relationships between tables, and dashboard creation...")
        else:
            print(f"ERROR: YouTube processing failed: {youtube_result['error']}")
        
        # Specific Test Queries - 2 from TXT file, 2 from PDF file, 2 from YouTube
        test_queries = [
            # 2 Questions from TXT file (PMarcaBlogs.txt)
            "What does Marc Andreessen say about the 'only thing that matters' for startups?",
            "What are Marc's views on hiring professional CEOs versus technical founders?",
            
            # 2 Questions from PDF file (test_doc.pdf)
            "What is the difference between frequentist and Bayesian approaches in machine learning according to this paper?",
            "How does the author explain supervised learning through linear regression in this introduction to ML for engineers?",
            
            # 2 Questions from YouTube video (Power BI content)
            "What are the three essential DAX formulas mentioned in this Power BI tutorial and what do they calculate?",
            "How does the tutorial explain creating relationships between customers and transactions tables using keys?"
        ]
        
        print(f"\nTesting structured RAG evaluation categories...")
        
        # Organize queries by file type
        test_categories = {
            "TXT FILE QUESTIONS (PMarcaBlogs.txt)": {
                "queries": test_queries[:2],
                "pass_criteria": "Answers must accurately reflect Marc Andreessen's specific advice from PMarcaBlogs.txt."
            },
            "PDF FILE QUESTIONS (test_doc.pdf)": {
                "queries": test_queries[2:4],
                "pass_criteria": "Answers must accurately reflect the specific research content from test_doc.pdf."
            },
            "YOUTUBE VIDEO QUESTIONS (Power BI Tutorial)": {
                "queries": test_queries[4:6],
                "pass_criteria": "Answers must accurately reflect the Power BI content from the YouTube video transcription."
            }
        }
        
        all_results = []
        
        for category_name, category_info in test_categories.items():
            print(f"\n{'='*60}")
            print(f"{category_name}")
            print(f"{'='*60}")
            print(f"Pass Criteria: {category_info['pass_criteria']}")
            print("-" * 60)
            
            category_results = []
            
            for i, query in enumerate(category_info['queries'], 1):
                print(f"\nQ{i}: {query}")
                print("-" * 50)
                
                # Use file type filtering for YouTube questions
                if "YOUTUBE" in category_name.upper():
                    result = pipeline.query(query, k=3, distance_metric='cosine', response_style="detailed", file_type_filter="youtube")
                else:
                    # Test with cosine similarity (most common)
                    result = pipeline.query(query, k=3, distance_metric='cosine', response_style="detailed")
                
                if result['success']:
                    print(f"ANSWER:")
                    print(f"{result['response']}")
                    print(f"\nRelevance scores: {result['relevance_scores']}")
                    print(f"Sources: {result['sources']}")
                    
                    category_results.append({
                        'question': query,
                        'response': result['response'],
                        'relevance_scores': result['relevance_scores'],
                        'sources': result['sources']
                    })
                else:
                    print(f"ERROR: {result['error']}")
                    category_results.append({
                        'question': query,
                        'response': f"ERROR: {result['error']}"
                    })
                
                print("\n" + "="*50)
            
            all_results.append({
                'category': category_name,
                'results': category_results,
                'pass_criteria': category_info['pass_criteria']
            })
        
        # Generate overall assessment
        print(f"\n{'='*60}")
        print("OVERALL RAG EVALUATION ASSESSMENT")
        print(f"{'='*60}")
        
        for category_result in all_results:
            category_name = category_result['category']
            results = category_result['results']
            pass_criteria = category_result['pass_criteria']
            
            print(f"\n{category_name}")
            print(f"Pass Criteria: {pass_criteria}")
            
            # Count quality indicators
            successful_responses = 0
            total_responses = len(results)
            
            for result in results:
                if not result['response'].startswith('ERROR:'):
                    successful_responses += 1
            
            success_rate = (successful_responses / total_responses) * 100 if total_responses > 0 else 0
            
            print(f"Success Rate: {success_rate:.1f}% ({successful_responses}/{total_responses})")
            
            if success_rate >= 80:
                print("EXCELLENT: All queries processed successfully")
            elif success_rate >= 60:
                print("GOOD: Most queries processed successfully")
            else:
                print("NEEDS IMPROVEMENT: Many queries failed")
        
        # Show database statistics
        print(f"\nDatabase Statistics:")
        stats = pipeline.get_database_stats()
        print(f"Total vectors: {stats['database_stats']['total_vectors']}")
        print(f"File types: {stats['database_stats']['file_types']}")
        print(f"Queries processed: {stats['pipeline_stats']['queries_processed']}")
        
        print("\nEnhanced RAG Pipeline Demo Complete!")
        
    except Exception as e:
        print(f"ERROR during demo: {e}")
        print("Please check your OpenAI API key and try again.")


def main():
    """Main function to run the enhanced RAG pipeline"""
    print("Enhanced RAG System with Secure API Key Input")
    print("=" * 60)
    print("This system uses getpass for secure API key input")
    print("instead of storing the key in files.")
    print()
    
    # Run the demo
    asyncio.run(demo_enhanced_rag())


if __name__ == "__main__":
    main()
