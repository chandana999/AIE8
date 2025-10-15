# evaluation_analysis.py

import time
import ast
import json
import pandas as pd
from langsmith.evaluation import evaluate as langsmith_evaluate
from ragas import EvaluationDataset, evaluate as ragas_evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness, FactualCorrectness, ResponseRelevancy, 
    ContextEntityRecall, ContextPrecision, ContextRecall
)
from langchain_openai import ChatOpenAI

# Storage for RAGAS
captured_results = {}

class CapturingChain:
    """Wrapper that captures results while LangSmith runs"""
    def __init__(self, chain, storage_key, dataset_df, delay_seconds=0):
        self.chain = chain
        self.storage_key = storage_key
        self.dataset_df = dataset_df
        self.delay_seconds = delay_seconds
        self.call_count = 0
        captured_results[storage_key] = []
    
    def invoke(self, inputs):
        if self.call_count > 0 and self.delay_seconds > 0:
            time.sleep(self.delay_seconds)
        
        # Run the chain
        output = self.chain.invoke(inputs)
        
        # Capture for RAGAS
        question = inputs["question"]
        matching_row = self.dataset_df[self.dataset_df["user_input"] == question]
        if not matching_row.empty:
            captured_results[self.storage_key].append({
                "user_input": question,
                "response": output["output"].content,
                "retrieved_contexts": [doc.page_content for doc in output["context"]],
                "reference": matching_row.iloc[0]["reference"],
                "reference_contexts": matching_row.iloc[0]["reference_contexts"]
            })
        
        self.call_count += 1
        return output

def setup_ragas_evaluation():
    """Setup RAGAS evaluation with metrics and config"""
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    custom_run_config = RunConfig(timeout=600, max_workers=4)
    
    print("✅ RAGAS evaluation setup complete")
    return evaluator_llm, custom_run_config

def evaluate_with_ragas(results, retriever_name, evaluator_llm, custom_run_config):
    """Evaluate a retriever's results with RAGAS metrics"""
    print(f"Evaluating {retriever_name}...")
    
    # Convert to RAGAS dataset
    ragas_dataset = EvaluationDataset.from_list(results)
    
    # Evaluate with BOTH generation AND retriever metrics
    scores = ragas_evaluate(
        dataset=ragas_dataset,
        metrics=[
            # Retriever-specific metrics (main focus for assignment)
            ContextPrecision(),       # How precise is retrieval?
            ContextRecall(),          # How complete is retrieval?
            ContextEntityRecall(),    # Are key entities retrieved?
            # Generation metrics (show overall quality)
            Faithfulness(),           # Is response faithful to context?
            FactualCorrectness(),     # Is response factually correct?
            ResponseRelevancy(),      # Is response relevant to question?
        ],
        llm=evaluator_llm,
        run_config=custom_run_config
    )
    
    print(f"{retriever_name} completed!")
    return scores

def evaluate_with_ragas_corrected(results, retriever_name, evaluator_llm, custom_run_config):
    """Evaluate a retriever's results with RAGAS metrics - corrected column names"""
    print(f"Evaluating {retriever_name}...")
    
    try:
        # Convert to RAGAS dataset format with correct column names
        ragas_data = []
        for result in results:
            # Fix reference_contexts - convert from string to list
            reference_contexts = result["reference_contexts"]
            if isinstance(reference_contexts, str):
                try:
                    # Try to parse as Python literal first
                    reference_contexts = ast.literal_eval(reference_contexts)
                except:
                    try:
                        # Try JSON parsing
                        reference_contexts = json.loads(reference_contexts)
                    except:
                        # If all else fails, create a single-item list
                        reference_contexts = [reference_contexts]
            
            # Ensure it's a list
            if not isinstance(reference_contexts, list):
                reference_contexts = [reference_contexts]
            
            # Fix retrieved_contexts - ensure it's a list
            retrieved_contexts = result["retrieved_contexts"]
            if not isinstance(retrieved_contexts, list):
                retrieved_contexts = [retrieved_contexts]
            
            # Create the data entry with CORRECT column names for RAGAS
            ragas_entry = {
                "user_input": result["user_input"],           # Correct column name
                "reference": result["reference"],             # Correct column name  
                "retrieved_contexts": retrieved_contexts,     # Correct column name
                "response": result["response"]                # Additional column for generation metrics
            }
            
            ragas_data.append(ragas_entry)
        
        print(f"📊 Sample data for {retriever_name}:")
        print(f"  Question: {ragas_data[0]['user_input'][:50]}...")
        print(f"  Retrieved contexts count: {len(ragas_data[0]['retrieved_contexts'])}")
        print(f"  Reference: {ragas_data[0]['reference'][:50]}...")
        print(f"  Response: {ragas_data[0]['response'][:50]}...")
        
        # Create RAGAS dataset
        ragas_dataset = EvaluationDataset.from_list(ragas_data)
        
        # Evaluate with retriever-specific metrics
        scores = ragas_evaluate(
            dataset=ragas_dataset,
            metrics=[
                ContextPrecision(),       # How precise is retrieval?
                ContextRecall(),          # How complete is retrieval?
                ContextEntityRecall(),    # Are key entities retrieved?
                Faithfulness(),           # Is response faithful to context?
                FactualCorrectness(),     # Is response factually correct?
                ResponseRelevancy(),      # Is response relevant to question?
            ],
            llm=evaluator_llm,
            run_config=custom_run_config
        )
        
        print(f"✅ {retriever_name} completed!")
        return scores
        
    except Exception as e:
        print(f"❌ Error evaluating {retriever_name}: {e}")
        print(f"Debug info:")
        print(f"  Results count: {len(results)}")
        if results:
            print(f"  Sample result keys: {list(results[0].keys())}")
        return None

def run_langsmith_evaluations(chains_dict, dataset, dataset_name="Synthetic Data for S09-Assignment"):
    """Run LangSmith evaluations for all retrievers"""
    print("🚀 Running LangSmith evaluations...")
    
    # Run LangSmith evaluations (chains run once, results captured)
    configs = [
        (chains_dict['naive_chain'], "naive-retriever", "Naive Retriever", 0),
        (chains_dict['bm25_chain'], "bm25-retriever", "BM25 Retriever", 0),
        (chains_dict['compression_chain'], "compression-retriever", "Compression Retriever", 7),
        (chains_dict['multi_query_chain'], "multiquery-retriever", "Multi-Query Retriever", 0),
        (chains_dict['parent_chain'], "parent-retriever", "Parent Retriever", 0),
        (chains_dict['ensemble_chain'], "ensemble-retriever", "Ensemble Retriever", 7),
        (chains_dict['semantic_chain'], "semantic-retriever", "Semantic Chunk Retriever", 0),
    ]
    
    langsmith_results = {}
    
    for chain, exp_name, display_name, delay in configs:
        print(f"Running {display_name}...")
        capturing_chain = CapturingChain(chain, display_name, dataset, delay)
        langsmith_results[exp_name] = langsmith_evaluate(
            capturing_chain.invoke,
            data=dataset_name,
            experiment_prefix=exp_name,
        )
        print(f"{display_name} completed!\n")
    
    print("✅ All LangSmith evaluations completed")
    return langsmith_results

def run_ragas_analysis(evaluator_llm, custom_run_config):
    """Run RAGAS analysis with captured results"""
    print("🔍 Running RAGAS analysis...")
    
    ragas_scores = {}
    for display_name, results in captured_results.items():
        if results and len(results) > 0:
            print(f"\n🔍 Processing {display_name} with {len(results)} results...")
            try:
                ragas_scores[display_name] = evaluate_with_ragas_corrected(results, display_name, evaluator_llm, custom_run_config)
            except Exception as e:
                print(f"❌ Failed to evaluate {display_name}: {e}")
                # Try with minimal metrics
                try:
                    print(f"🔄 Trying with minimal metrics for {display_name}...")
                    
                    # Create minimal dataset with correct column names
                    simple_data = []
                    for result in results:
                        # Force convert everything to proper types with correct column names
                        simple_data.append({
                            "user_input": str(result["user_input"]),
                            "reference": str(result["reference"]),
                            "retrieved_contexts": [str(ctx) for ctx in result["retrieved_contexts"]] if isinstance(result["retrieved_contexts"], list) else [str(result["retrieved_contexts"])],
                            "response": str(result["response"])
                        })
                    
                    ragas_dataset = EvaluationDataset.from_list(simple_data)
                    scores = ragas_evaluate(
                        dataset=ragas_dataset,
                        metrics=[ContextPrecision(), Faithfulness()],  # Just 2 metrics
                        llm=evaluator_llm,
                        run_config=custom_run_config
                    )
                    ragas_scores[display_name] = scores
                    print(f"✅ {display_name} completed with minimal metrics")
                    
                except Exception as e2:
                    print(f"❌ Even minimal evaluation failed for {display_name}: {e2}")
        else:
            print(f"⚠️ No results captured for {display_name}")
    
    print("✅ RAGAS analysis completed")
    return ragas_scores

def generate_results_tables(ragas_scores):
    """Generate results tables from RAGAS scores"""
    print("📊 Generating results tables...")
    
    # TABLE 1: Aggregated Average Scores per Retriever (TRANSPOSED)
    all_results = []
    for name in ["Naive Retriever", "BM25 Retriever", "Compression Retriever", 
                 "Multi-Query Retriever", "Parent Retriever", "Ensemble Retriever", 
                 "Semantic Chunk Retriever"]:
        if name in ragas_scores:
            df = ragas_scores[name].to_pandas()
            df.insert(0, 'retriever', name)
            all_results.append(df)
    
    if all_results:
        results_comparison = pd.concat(all_results, ignore_index=True)
        
        # Get metric columns (exclude metadata)
        metric_columns = [col for col in results_comparison.columns 
                          if col not in ['retriever', 'user_input', 'retrieved_contexts', 
                                         'reference', 'reference_contexts', 'response']]
        
        # Calculate mean scores per retriever
        aggregated_scores = results_comparison.groupby('retriever')[metric_columns].mean()
        
        # Reorder rows
        aggregated_scores = aggregated_scores.reindex(["Naive Retriever", "BM25 Retriever", 
                                                        "Compression Retriever", "Multi-Query Retriever", 
                                                        "Parent Retriever", "Ensemble Retriever", 
                                                        "Semantic Chunk Retriever"])
        
        # TRANSPOSE: metrics as rows, retrievers as columns
        aggregated_scores = aggregated_scores.T
        
        print("📊 TABLE 1: Average Scores per Retriever")
        print("=" * 120)
        with pd.option_context('display.max_columns', None, 
                               'display.width', None,
                               'display.float_format', '{:.4f}'.format):
            print(aggregated_scores)
    else:
        print("⚠️ No RAGAS results available to display")
        aggregated_scores = None
        metric_columns = []
    
    print("✅ Results tables generated")
    return aggregated_scores, metric_columns, all_results

def generate_final_summary(aggregated_scores, metric_columns):
    """Generate final summary and recommendations"""
    print("🏆 FINAL SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)
    
    if aggregated_scores is not None:
        # Calculate overall performance
        performance_summary = aggregated_scores.mean(axis=1).sort_values(ascending=False)
        
        print("\n📊 OVERALL PERFORMANCE RANKING:")
        print("-" * 40)
        for i, (metric, score) in enumerate(performance_summary.items(), 1):
            print(f"{i:2d}. {metric}: {score:.4f}")
        
        # Find best retriever for each metric
        print("\n🏆 BEST RETRIEVER BY METRIC:")
        print("-" * 40)
        for metric in metric_columns:
            best_retriever = aggregated_scores.loc[metric].idxmax()
            best_score = aggregated_scores.loc[metric].max()
            print(f"{metric}: {best_retriever} ({best_score:.4f})")
        
        # Overall best retriever
        retriever_avg_scores = aggregated_scores.mean(axis=0).sort_values(ascending=False)
        best_overall = retriever_avg_scores.index[0]
        best_score = retriever_avg_scores.iloc[0]
        
        print(f"\n🥇 OVERALL BEST RETRIEVER: {best_overall}")
        print(f"📈 Average Score: {best_score:.4f}")
    
    print(f"\n✅ Activity 1 - Advanced Retrieval Evaluation Completed!")
    print(f"📁 Results saved and analysis complete")
    print(f"🔗 Check LangSmith dashboard for detailed traces: https://smith.langchain.com/")

def run_langsmith_evaluations_only(chains_dict, dataset, dataset_name="Synthetic Data for S09-Assignment"):
    """Run only LangSmith evaluations"""
    print("🚀 Starting LangSmith Evaluations...")
    
    # Setup RAGAS evaluation
    evaluator_llm, custom_run_config = setup_ragas_evaluation()
    
    # Run LangSmith evaluations
    langsmith_results = run_langsmith_evaluations(chains_dict, dataset, dataset_name)
    
    print("✅ LangSmith evaluations completed")
    return evaluator_llm, custom_run_config, langsmith_results

def run_ragas_analysis_only(evaluator_llm, custom_run_config):
    """Run only RAGAS analysis"""
    print("🔍 Starting RAGAS Analysis...")
    
    # Run RAGAS analysis
    ragas_scores = run_ragas_analysis(evaluator_llm, custom_run_config)
    
    print("✅ RAGAS analysis completed")
    return ragas_scores

def generate_results_tables_only(ragas_scores):
    """Generate results tables only"""
    print("📊 Generating Results Tables...")
    
    # Generate results tables
    aggregated_scores, metric_columns, all_results = generate_results_tables(ragas_scores)
    
    print("✅ Results tables completed")
    return aggregated_scores, metric_columns, all_results

def generate_final_summary_only(aggregated_scores, metric_columns):
    """Generate final summary only"""
    print("📊 Generating Final Summary...")
    
    # Generate final summary
    generate_final_summary(aggregated_scores, metric_columns)
    
    print("✅ Final summary completed")

def generate_results_and_summary_only(ragas_scores):
    """Generate results tables and final summary"""
    print("📊 Generating Results and Summary...")
    
    # Generate results tables
    aggregated_scores, metric_columns, all_results = generate_results_tables(ragas_scores)
    
    # Generate final summary
    generate_final_summary(aggregated_scores, metric_columns)
    
    print("✅ Results and summary completed")
    return aggregated_scores, metric_columns, all_results

def main_evaluation_analysis(chains_dict, dataset, dataset_name="Synthetic Data for S09-Assignment"):
    """Main function to run complete evaluation analysis - uses the split functions"""
    print("🚀 Starting Complete Evaluation Analysis...")
    
    # Step 1: Run LangSmith evaluations
    evaluator_llm, custom_run_config, langsmith_results = run_langsmith_evaluations_only(chains_dict, dataset, dataset_name)
    
    # Step 2: Run RAGAS analysis
    ragas_scores = run_ragas_analysis_only(evaluator_llm, custom_run_config)
    
    # Step 3: Generate results tables and summary
    aggregated_scores, metric_columns, all_results = generate_results_and_summary_only(ragas_scores)
    
    print("🎉 Complete Evaluation Analysis Finished!")
    
    return {
        'langsmith_results': langsmith_results,
        'ragas_scores': ragas_scores,
        'aggregated_scores': aggregated_scores,
        'metric_columns': metric_columns,
        'all_results': all_results,
        'captured_results': captured_results
    }
	