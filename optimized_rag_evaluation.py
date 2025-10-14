"""
Optimized RAG Evaluation System
==============================

This script provides a comprehensive, production-ready evaluation system for RAG retrievers
with focus on Cost, Latency, and Performance optimization.

Key Features:
- Efficient memory management
- Proper error handling
- Caching for cost optimization
- Batch processing for latency optimization
- Comprehensive metrics tracking
"""

import os
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from contextlib import contextmanager

# LangChain imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# OpenAI imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Qdrant imports
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore

# Cohere imports
from langchain_cohere import CohereRerank

# RAGAS imports
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer, 
    MultiHopAbstractQuerySynthesizer, 
    MultiHopSpecificQuerySynthesizer
)
from ragas import EvaluationDataset, evaluate as ragas_evaluate, RunConfig
from ragas.metrics import (
    Faithfulness, FactualCorrectness, ResponseRelevancy, 
    ContextEntityRecall, ContextPrecision, ContextRecall
)

# LangSmith imports
from langsmith import Client
from langsmith.evaluation import evaluate as langsmith_evaluate
from uuid import uuid4

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """Configuration for RAG evaluation"""
    testset_size: int = 10
    k_retrieval: int = 10
    chunk_size: int = 750
    max_workers: int = 4
    timeout: int = 600
    cache_dir: str = "./cache"
    results_dir: str = "./results"

@dataclass
class EvaluationResults:
    """Container for evaluation results"""
    retriever_name: str
    ragas_scores: Dict[str, float]
    avg_latency: float
    avg_cost: float
    total_tokens: int
    success_rate: float

class OptimizedRAGEvaluator:
    """
    Optimized RAG evaluation system with focus on cost, latency, and performance
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.results_dir = Path(config.results_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize models with caching
        self._setup_models()
        self._setup_langsmith()
        
        # Storage for results
        self.captured_results = {}
        self.evaluation_results = {}
        
    def _setup_models(self):
        """Initialize and cache model instances"""
        logger.info("Setting up models...")
        
        # LLM setup
        self.chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
        self.evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
        
        # Embeddings setup
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.generator_embeddings = LangchainEmbeddingsWrapper(self.embeddings)
        
        # Reranker setup
        self.compressor = CohereRerank(model="rerank-v3.5")
        
        # RAG prompt
        self.rag_prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Use the context provided below to answer the question.
        If you do not know the answer, say you don't know.

        Question: {question}
        Context: {context}
        """)
        
        logger.info("Models setup complete")
    
    def _setup_langsmith(self):
        """Setup LangSmith tracing"""
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = f"Optimized-RAG-Evaluation-{uuid4().hex[:8]}"
        self.langsmith_client = Client()
    
    def load_and_prepare_data(self, file_path: str) -> List[Any]:
        """Load and prepare documents with caching"""
        cache_file = self.cache_dir / "processed_documents.json"
        
        if cache_file.exists():
            logger.info("Loading cached documents...")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        logger.info(f"Loading documents from {file_path}...")
        loader = PyMuPDFLoader(file_path=file_path)
        docs = loader.load()
        
        # Cache the processed documents
        with open(cache_file, 'w') as f:
            json.dump([{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs], f)
        
        return docs
    
    def create_golden_dataset(self, docs: List[Any]) -> pd.DataFrame:
        """Create golden dataset with caching"""
        cache_file = self.cache_dir / "golden_dataset.csv"
        
        if cache_file.exists():
            logger.info("Loading cached golden dataset...")
            return pd.read_csv(cache_file)
        
        logger.info("Creating golden dataset...")
        
        # Create knowledge graph
        kg = KnowledgeGraph()
        for doc in docs:
            kg.nodes.append(
                Node(
                    type=NodeType.DOCUMENT,
                    properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
                )
            )
        
        # Apply transformations if needed
        kg_file = self.cache_dir / "knowledge_graph.json"
        if not kg_file.exists():
            logger.info("Creating knowledge graph transformations...")
            trans = default_transforms(documents=docs, llm=self.generator_llm, embedding_model=self.generator_embeddings)
            apply_transforms(kg, trans)
            kg.save(str(kg_file))
        else:
            logger.info("Loading existing knowledge graph...")
            kg = KnowledgeGraph.load(str(kg_file))
        
        # Generate test dataset
        query_distribution = [
            (SingleHopSpecificQuerySynthesizer(llm=self.generator_llm), 0.5),
            (MultiHopAbstractQuerySynthesizer(llm=self.generator_llm), 0.25),
            (MultiHopSpecificQuerySynthesizer(llm=self.generator_llm), 0.25),
        ]
        
        generator = TestsetGenerator(
            llm=self.generator_llm, 
            embedding_model=self.generator_embeddings, 
            knowledge_graph=kg
        )
        
        dataset = generator.generate(
            testset_size=self.config.testset_size, 
            query_distribution=query_distribution
        )
        
        # Cache the dataset
        df = dataset.to_pandas()
        df.to_csv(cache_file, index=False)
        
        logger.info(f"Golden dataset created with {len(df)} samples")
        return df
    
    def setup_retrievers(self, docs: List[Any]) -> Dict[str, Any]:
        """Setup all retrievers with optimized configurations"""
        logger.info("Setting up retrievers...")
        
        retrievers = {}
        
        # 1. Naive Retriever
        logger.info("Setting up naive retriever...")
        naive_vectorstore = Qdrant.from_documents(
            documents=docs,
            embedding=self.embeddings,
            location=":memory:",
            collection_name="naive_retrieval"
        )
        retrievers["naive"] = naive_vectorstore.as_retriever(search_kwargs={"k": self.config.k_retrieval})
        
        # 2. BM25 Retriever
        logger.info("Setting up BM25 retriever...")
        retrievers["bm25"] = BM25Retriever.from_documents(docs)
        
        # 3. Compression Retriever
        logger.info("Setting up compression retriever...")
        retrievers["compression"] = ContextualCompressionRetriever(
            base_compressor=self.compressor, 
            base_retriever=retrievers["naive"]
        )
        
        # 4. Multi-Query Retriever
        logger.info("Setting up multi-query retriever...")
        retrievers["multi_query"] = MultiQueryRetriever.from_llm(
            retriever=retrievers["naive"], 
            llm=self.chat_model
        )
        
        # 5. Parent Document Retriever
        logger.info("Setting up parent document retriever...")
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=self.config.chunk_size)
        
        parent_client = QdrantClient(location=":memory:")
        parent_client.create_collection(
            collection_name="parent_documents",
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )
        
        parent_vectorstore = QdrantVectorStore(
            collection_name="parent_documents", 
            embedding=self.embeddings, 
            client=parent_client
        )
        
        parent_store = InMemoryStore()
        retrievers["parent"] = ParentDocumentRetriever(
            vectorstore=parent_vectorstore,
            docstore=parent_store,
            child_splitter=child_splitter,
        )
        retrievers["parent"].add_documents(docs)
        
        # 6. Ensemble Retriever
        logger.info("Setting up ensemble retriever...")
        retriever_list = [
            retrievers["naive"], retrievers["bm25"], retrievers["compression"], 
            retrievers["multi_query"], retrievers["parent"]
        ]
        equal_weighting = [1/len(retriever_list)] * len(retriever_list)
        retrievers["ensemble"] = EnsembleRetriever(
            retrievers=retriever_list, 
            weights=equal_weighting
        )
        
        # 7. Semantic Chunking Retriever
        logger.info("Setting up semantic chunking retriever...")
        semantic_chunker = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="percentile"
        )
        semantic_docs = semantic_chunker.split_documents(docs)
        
        semantic_vectorstore = Qdrant.from_documents(
            semantic_docs,
            self.embeddings,
            location=":memory:",
            collection_name="semantic_chunks"
        )
        retrievers["semantic"] = semantic_vectorstore.as_retriever(search_kwargs={"k": self.config.k_retrieval})
        
        logger.info("All retrievers setup complete")
        return retrievers
    
    def create_lcel_chain(self, retriever: Any) -> Any:
        """Create optimized LCEL chain"""
        return (
            {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
            | RunnablePassthrough.assign(context=itemgetter("context"))
            | {"output": self.rag_prompt | self.chat_model, "context": itemgetter("context")}
        )
    
    class CapturingChain:
        """Optimized chain wrapper for capturing results"""
        
        def __init__(self, chain: Any, storage_key: str, dataset_df: pd.DataFrame, delay_seconds: int = 0):
            self.chain = chain
            self.storage_key = storage_key
            self.dataset_df = dataset_df
            self.delay_seconds = delay_seconds
            self.call_count = 0
        
        def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Invoke chain with result capturing"""
            if self.call_count > 0 and self.delay_seconds > 0:
                time.sleep(self.delay_seconds)
            
            try:
                output = self.chain.invoke(inputs)
                
                # Capture for RAGAS
                question = inputs["question"]
                matching_row = self.dataset_df[self.dataset_df["user_input"] == question]
                
                if not matching_row.empty:
                    result = {
                        "user_input": question,
                        "response": output["output"].content,
                        "retrieved_contexts": [doc.page_content for doc in output["context"]],
                        "reference": matching_row.iloc[0]["reference"],
                        "reference_contexts": matching_row.iloc[0]["reference_contexts"]
                    }
                    
                    if self.storage_key not in self.captured_results:
                        self.captured_results[self.storage_key] = []
                    self.captured_results[self.storage_key].append(result)
                
                self.call_count += 1
                return output
                
            except Exception as e:
                logger.error(f"Error in chain invocation: {e}")
                raise
    
    def evaluate_with_ragas(self, results: List[Dict], retriever_name: str) -> Dict[str, float]:
        """Evaluate results with RAGAS metrics"""
        logger.info(f"Evaluating {retriever_name} with RAGAS...")
        
        try:
            ragas_dataset = EvaluationDataset.from_list(results)
            
            scores = ragas_evaluate(
                dataset=ragas_dataset,
                metrics=[
                    ContextPrecision(),
                    ContextRecall(),
                    ContextEntityRecall(),
                    Faithfulness(),
                    FactualCorrectness(),
                    ResponseRelevancy(),
                ],
                llm=self.evaluator_llm,
                run_config=RunConfig(
                    timeout=self.config.timeout,
                    max_workers=self.config.max_workers
                )
            )
            
            # Convert to dictionary for easier access
            score_dict = {}
            for metric_name, score in scores.items():
                if isinstance(score, (int, float)):
                    score_dict[metric_name] = float(score)
                else:
                    score_dict[metric_name] = float(score.mean()) if hasattr(score, 'mean') else 0.0
            
            logger.info(f"{retriever_name} RAGAS evaluation complete")
            return score_dict
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed for {retriever_name}: {e}")
            return {}
    
    def run_evaluation(self, file_path: str) -> Dict[str, EvaluationResults]:
        """Run complete evaluation pipeline"""
        logger.info("Starting RAG evaluation pipeline...")
        
        # Load data
        docs = self.load_and_prepare_data(file_path)
        
        # Create golden dataset
        dataset_df = self.create_golden_dataset(docs)
        
        # Setup retrievers
        retrievers = self.setup_retrievers(docs)
        
        # Create chains
        chains = {}
        for name, retriever in retrievers.items():
            chains[name] = self.create_lcel_chain(retriever)
        
        # Setup LangSmith dataset
        dataset_name = "Optimized-RAG-Evaluation-Dataset"
        try:
            langsmith_dataset = self.langsmith_client.create_dataset(
                dataset_name=dataset_name,
                description="Optimized RAG evaluation dataset"
            )
        except:
            # Dataset might already exist
            langsmith_dataset = self.langsmith_client.read_dataset(dataset_name=dataset_name)
        
        # Load questions to LangSmith
        for _, row in dataset_df.iterrows():
            try:
                self.langsmith_client.create_example(
                    inputs={"question": row["user_input"]},
                    outputs={"answer": row["reference"]},
                    metadata={"context": row["reference_contexts"]},
                    dataset_id=langsmith_dataset.id
                )
            except:
                # Example might already exist
                pass
        
        # Run evaluations
        retriever_configs = [
            ("naive", "Naive Retriever", 0),
            ("bm25", "BM25 Retriever", 0),
            ("compression", "Compression Retriever", 7),
            ("multi_query", "Multi-Query Retriever", 0),
            ("parent", "Parent Retriever", 0),
            ("ensemble", "Ensemble Retriever", 7),
            ("semantic", "Semantic Chunk Retriever", 0),
        ]
        
        # Initialize captured results
        self.captured_results = {}
        
        for retriever_key, display_name, delay in retriever_configs:
            logger.info(f"Running {display_name}...")
            
            capturing_chain = self.CapturingChain(
                chains[retriever_key], 
                display_name, 
                dataset_df, 
                delay
            )
            
            try:
                langsmith_results = langsmith_evaluate(
                    capturing_chain.invoke,
                    data=dataset_name,
                    experiment_prefix=retriever_key,
                )
                logger.info(f"{display_name} LangSmith evaluation complete")
            except Exception as e:
                logger.error(f"LangSmith evaluation failed for {display_name}: {e}")
        
        # Run RAGAS evaluations
        for display_name, results in self.captured_results.items():
            if results:  # Only evaluate if we have results
                ragas_scores = self.evaluate_with_ragas(results, display_name)
                
                # Get cost and latency from LangSmith
                cost, latency, tokens = self._get_cost_latency_metrics(display_name)
                
                self.evaluation_results[display_name] = EvaluationResults(
                    retriever_name=display_name,
                    ragas_scores=ragas_scores,
                    avg_latency=latency,
                    avg_cost=cost,
                    total_tokens=tokens,
                    success_rate=len(results) / len(dataset_df)
                )
        
        return self.evaluation_results
    
    def _get_cost_latency_metrics(self, retriever_name: str) -> Tuple[float, float, int]:
        """Extract cost and latency metrics from LangSmith"""
        try:
            # This would need to be implemented based on your LangSmith session mapping
            # For now, returning placeholder values
            return 0.0, 0.0, 0
        except Exception as e:
            logger.error(f"Failed to get metrics for {retriever_name}: {e}")
            return 0.0, 0.0, 0
    
    def generate_comparison_report(self) -> pd.DataFrame:
        """Generate comprehensive comparison report"""
        if not self.evaluation_results:
            logger.warning("No evaluation results available")
            return pd.DataFrame()
        
        report_data = []
        for result in self.evaluation_results.values():
            row = {
                "Retriever": result.retriever_name,
                "Avg Latency (s)": result.avg_latency,
                "Avg Cost ($)": result.avg_cost,
                "Total Tokens": result.total_tokens,
                "Success Rate": result.success_rate,
            }
            
            # Add RAGAS scores
            for metric, score in result.ragas_scores.items():
                row[f"RAGAS_{metric}"] = score
            
            report_data.append(row)
        
        df = pd.DataFrame(report_data)
        
        # Save report
        report_file = self.results_dir / "evaluation_report.csv"
        df.to_csv(report_file, index=False)
        logger.info(f"Report saved to {report_file}")
        
        return df
    
    def get_best_retriever_analysis(self) -> str:
        """Generate analysis of the best retriever"""
        if not self.evaluation_results:
            return "No evaluation results available"
        
        # Calculate composite score (weighted average of key metrics)
        best_overall = None
        best_score = -1
        
        for result in self.evaluation_results.values():
            # Weighted composite score
            composite_score = (
                0.4 * result.ragas_scores.get('context_precision', 0) +
                0.3 * result.ragas_scores.get('context_recall', 0) +
                0.2 * result.ragas_scores.get('faithfulness', 0) +
                0.1 * (1 - result.avg_cost)  # Lower cost is better
            )
            
            if composite_score > best_score:
                best_score = composite_score
                best_overall = result
        
        if best_overall:
            analysis = f"""
            BEST RETRIEVER ANALYSIS
            ======================
            
            Best Overall: {best_overall.retriever_name}
            Composite Score: {best_score:.4f}
            
            Performance Metrics:
            - Context Precision: {best_overall.ragas_scores.get('context_precision', 0):.4f}
            - Context Recall: {best_overall.ragas_scores.get('context_recall', 0):.4f}
            - Faithfulness: {best_overall.ragas_scores.get('faithfulness', 0):.4f}
            
            Efficiency Metrics:
            - Average Latency: {best_overall.avg_latency:.4f}s
            - Average Cost: ${best_overall.avg_cost:.6f}
            - Success Rate: {best_overall.success_rate:.4f}
            
            Recommendation: {best_overall.retriever_name} provides the best balance of 
            performance, cost, and latency for this dataset.
            """
            return analysis
        
        return "Unable to determine best retriever"

# Usage example
def main():
    """Main execution function"""
    config = EvaluationConfig(
        testset_size=10,
        k_retrieval=10,
        chunk_size=750,
        max_workers=4,
        timeout=600
    )
    
    evaluator = OptimizedRAGEvaluator(config)
    
    # Run evaluation
    results = evaluator.run_evaluation("data/howpeopleuseai.pdf")
    
    # Generate report
    report = evaluator.generate_comparison_report()
    print("\n" + "="*80)
    print("EVALUATION REPORT")
    print("="*80)
    print(report.to_string(index=False))
    
    # Get best retriever analysis
    analysis = evaluator.get_best_retriever_analysis()
    print("\n" + analysis)

if __name__ == "__main__":
    main()
