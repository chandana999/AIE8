"""
Simple script to run the optimized RAG evaluation
"""

import os
from optimized_rag_evaluation import OptimizedRAGEvaluator, EvaluationConfig

def main():
    # Set up API keys
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = input("Enter your OpenAI API Key: ")
    
    if not os.getenv("COHERE_API_KEY"):
        os.environ["COHERE_API_KEY"] = input("Enter your Cohere API Key: ")
    
    if not os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = input("Enter your LangSmith API Key: ")
    
    # Configuration
    config = EvaluationConfig(
        testset_size=10,  # Adjust based on your needs
        k_retrieval=10,
        chunk_size=750,
        max_workers=4,
        timeout=600
    )
    
    # Initialize evaluator
    evaluator = OptimizedRAGEvaluator(config)
    
    # Run evaluation
    print("🚀 Starting RAG evaluation...")
    results = evaluator.run_evaluation("data/howpeopleuseai.pdf")
    
    # Generate and display report
    print("\n📊 Generating evaluation report...")
    report = evaluator.generate_comparison_report()
    
    print("\n" + "="*100)
    print("📈 COMPREHENSIVE EVALUATION REPORT")
    print("="*100)
    print(report.to_string(index=False))
    
    # Best retriever analysis
    print("\n" + "="*100)
    print("🏆 BEST RETRIEVER ANALYSIS")
    print("="*100)
    analysis = evaluator.get_best_retriever_analysis()
    print(analysis)
    
    print("\n✅ Evaluation complete! Check the 'results/' directory for detailed reports.")

if __name__ == "__main__":
    main()
