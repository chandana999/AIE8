# advanced_retrieval_setup.py

import os
import getpass
from uuid import uuid4
from langsmith import Client
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereRerank
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore

def setup_langsmith(dataset, dataset_name="Synthetic Data for S09-Assignment"):
    """Setup LangSmith with getpass for API key - exact match to notebook"""
    
    # Enable Langchain tracing (langSmith)
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    # Get LangSmith API key using getpass
    langsmith_key = getpass.getpass("Enter your LangSmith API Key: ")

    # Set the API key
    os.environ["LANGCHAIN_API_KEY"] = langsmith_key

    # Create langsmith project
    os.environ["LANGCHAIN_PROJECT"] = f"AIM - S09-Assignment - {uuid4().hex[0:8]}"

    # Create the DS on LangSmith and setting the Client
    client = Client()

    try:
        langsmith_dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"📂 Using existing LangSmith dataset: {dataset_name}")
    except:
        langsmith_dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="SD for Retrievers"
        )
        print(f"📂 Created new LangSmith dataset: {dataset_name}")

    # Load questions to LangSmith (only if dataset is empty)
    existing_examples = list(client.list_examples(dataset_id=langsmith_dataset.id))
    
    if len(existing_examples) == 0:
        print("📝 Loading questions to LangSmith...")
        for data_row in dataset.iterrows():
            client.create_example(
                inputs={
                    "question": data_row[1]["user_input"]
                },
                outputs={
                    "answer": data_row[1]["reference"]
                },
                metadata={
                    "context": data_row[1]["reference_contexts"]
                },
                dataset_id=langsmith_dataset.id
            )
        print(f"✅ Loaded {len(dataset)} questions to LangSmith")
    else:
        print(f"📂 LangSmith dataset already has {len(existing_examples)} examples - skipping load")

    print("✅ LangSmith setup complete")
    return client, langsmith_dataset

def load_pdf_data_and_components(file_path='data/howpeopleuseai.pdf'):
    """Load PDF data and initialize shared components"""
    print("📄 Loading PDF data...")
    
    # Load PDF data
    loader = PyMuPDFLoader(file_path=file_path)
    pdf_docs = loader.load()
    print(f"✅ Loaded {len(pdf_docs)} documents from PDF")
    
    # Initialize shared components
    pdf_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=750)
    
    print("✅ Components initialized")
    return pdf_docs, pdf_embeddings, chat_model, child_splitter

def create_all_retrievers(pdf_docs, pdf_embeddings, chat_model, child_splitter):
    """Create all retrieval strategies"""
    print("🔧 Creating all retrieval strategies...")
    
    # Create vector store
    pdf_vectorstore = Qdrant.from_documents(
        documents=pdf_docs,
        embedding=pdf_embeddings,
        location=":memory:",
        collection_name="PDF_Synthetic_Questions"
    )
    
    # 1. Naive Retriever
    pdf_naive_retriever = pdf_vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # 2. BM25 Retriever
    pdf_bm25_retriever = BM25Retriever.from_documents(pdf_docs)
    
    # 3. Compression Retriever
    compressor = CohereRerank(model="rerank-v3.5")
    pdf_compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=pdf_naive_retriever
    )
    
    # 4. Multi-Query Retriever
    pdf_multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=pdf_naive_retriever, 
        llm=chat_model
    )
    
    # 5. Parent Document Retriever
    pdf_parent_docs = pdf_docs
    pdf_parent_client = QdrantClient(location=":memory:")
    pdf_parent_client.create_collection(
        collection_name="pdf_full_documents",
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
    )
    
    pdf_parent_document_vectorstore = QdrantVectorStore(
        collection_name="pdf_full_documents", 
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"), 
        client=pdf_parent_client
    )
    
    pdf_parent_store = InMemoryStore()
    pdf_parent_retriever = ParentDocumentRetriever(
        vectorstore=pdf_parent_document_vectorstore,
        docstore=pdf_parent_store,
        child_splitter=child_splitter,
    )
    pdf_parent_retriever.add_documents(pdf_parent_docs, ids=None)
    
    # 6. Ensemble Retriever
    pdf_retriever_list = [pdf_naive_retriever, pdf_bm25_retriever, pdf_compression_retriever, pdf_multi_query_retriever, pdf_parent_retriever]
    equal_weighting = [1/len(pdf_retriever_list)] * len(pdf_retriever_list)
    pdf_ensemble_retriever = EnsembleRetriever(retrievers=pdf_retriever_list, weights=equal_weighting)
    
    # 7. Semantic Chunking
    semantic_chunker = SemanticChunker(
        pdf_embeddings,
        breakpoint_threshold_type="percentile"
    )
    pdf_semantic_documents = semantic_chunker.split_documents(pdf_docs)
    
    pdf_semantic_vectorstore = Qdrant.from_documents(
        pdf_semantic_documents,
        pdf_embeddings,
        location=":memory:",
        collection_name="Synthetic_PDF_Data_Semantic_Chunks"
    )
    pdf_semantic_retriever = pdf_semantic_vectorstore.as_retriever(search_kwargs={"k": 10})
    
    print("✅ All retrievers created")
    
    return {
        'naive': pdf_naive_retriever,
        'bm25': pdf_bm25_retriever,
        'compression': pdf_compression_retriever,
        'multi_query': pdf_multi_query_retriever,
        'parent': pdf_parent_retriever,
        'ensemble': pdf_ensemble_retriever,
        'semantic': pdf_semantic_retriever,
        'vectorstore': pdf_vectorstore
    }

def create_rag_chains(retrievers_dict, chat_model):
    """Create RAG chains for all retrievers"""
    print("🔗 Creating RAG template and chains...")
    
    # RAG template
    RAG_TEMPLATE = """You are a helpful and kind assistant. Use the context provided below to answer the question.

If you do not know the answer, or are unsure, say you don't know.

Query:
{question}

Context:
{context}"""
    
    # Create prompt template
    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    
    # LCEL RAG Chain Function
    def lcel_chain(retriever):
        return (
            {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
            | RunnablePassthrough.assign(context=itemgetter("context"))
            | {"output": rag_prompt | chat_model, "context": itemgetter("context")}
        )
    
    # Create chains for all retrievers
    chains = {}
    for name, retriever in retrievers_dict.items():
        if name != 'vectorstore':  # Skip vectorstore
            chains[f"{name}_chain"] = lcel_chain(retriever)
    
    print("✅ RAG chains created for all retrievers")
    return chains, rag_prompt

def load_questions_to_langsmith(client, dataset, langsmith_dataset):
    """Load questions from dataset to LangSmith"""
    print("📝 Loading questions to LangSmith...")
    
    for data_row in dataset.iterrows():
        client.create_example(
            inputs={"question": data_row[1]["user_input"]},
            outputs={"answer": data_row[1]["reference"]},
            metadata={"context": data_row[1]["reference_contexts"]},
            dataset_id=langsmith_dataset.id
        )
    
    print("✅ Questions loaded to LangSmith")

def main_setup(dataset, dataset_name="Synthetic Data for S09-Assignment"):
    """Main function to set up complete Advanced Retrieval system"""
    print("🚀 Starting Advanced Retrieval Setup...")
    
    # Step 1: Setup LangSmith
    client, langsmith_dataset = setup_langsmith(dataset, dataset_name)
    
    # Step 2: Load PDF data and components
    pdf_docs, pdf_embeddings, chat_model, child_splitter = load_pdf_data_and_components()
    
    # Step 3: Create all retrievers
    retrievers = create_all_retrievers(pdf_docs, pdf_embeddings, chat_model, child_splitter)
    
    # Step 4: Create RAG chains
    chains, rag_prompt = create_rag_chains(retrievers, chat_model)
    
    print("🎉 Advanced Retrieval Setup Complete!")
    
    return {
        'client': client,
        'langsmith_dataset': langsmith_dataset,
        'pdf_docs': pdf_docs,
        'pdf_embeddings': pdf_embeddings,
        'chat_model': chat_model,
        'child_splitter': child_splitter,
        'retrievers': retrievers,
        'chains': chains,
        'rag_prompt': rag_prompt
    }