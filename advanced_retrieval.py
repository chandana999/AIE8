def load_log_data_and_components():
    """Load log incident data and initialize shared components"""
    print("📄 Loading log incident data...")
    
    # Use your existing log documents
    log_docs = all_knowledge_documents  # Your incident documents
    print(f"✅ Using {len(log_docs)} log incident documents")
    
    # Initialize shared components
    log_embeddings = embedding_model  # Your existing embedding model
    chat_model = generator_llm        # Your existing LLM
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=750)
    
    print("✅ Components initialized")
    return log_docs, log_embeddings, chat_model, child_splitter

def create_all_log_retrievers(log_docs, log_embeddings, chat_model, child_splitter):
    """Create all retrieval strategies for log analytics"""
    print("🔧 Creating all retrieval strategies for log analytics...")
    
    # Create vector store
    log_vectorstore = Qdrant.from_documents(
        documents=log_docs,
        embedding=log_embeddings,
        location=":memory:",
        collection_name="Log_Incident_Knowledge_Base"
    )
    
    # 1. Naive Retriever
    log_naive_retriever = log_vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # 2. BM25 Retriever
    log_bm25_retriever = BM25Retriever.from_documents(log_docs)
    
    # 3. Compression Retriever (optional - skip if no Cohere)
    try:
        compressor = CohereRerank(model="rerank-v3.5")
        log_compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=log_naive_retriever
        )
    except:
        print("⚠️ Cohere not available, skipping compression retriever")
        log_compression_retriever = log_naive_retriever
    
    # 4. Multi-Query Retriever
    log_multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=log_naive_retriever, 
        llm=chat_model
    )
    
    # 5. Parent Document Retriever
    log_parent_docs = log_docs
    log_parent_client = QdrantClient(location=":memory:")
    log_parent_client.create_collection(
        collection_name="log_full_documents",
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
    )
    
    log_parent_document_vectorstore = QdrantVectorStore(
        collection_name="log_full_documents", 
        embedding=log_embeddings, 
        client=log_parent_client
    )
    
    log_parent_store = InMemoryStore()
    log_parent_retriever = ParentDocumentRetriever(
        vectorstore=log_parent_document_vectorstore,
        docstore=log_parent_store,
        child_splitter=child_splitter,
    )
    log_parent_retriever.add_documents(log_parent_docs, ids=None)
    
    # 6. Ensemble Retriever (only include available retrievers)
    available_retrievers = [log_naive_retriever, log_bm25_retriever, log_multi_query_retriever, log_parent_retriever]
    if log_compression_retriever != log_naive_retriever:
        available_retrievers.append(log_compression_retriever)
    
    equal_weighting = [1/len(available_retrievers)] * len(available_retrievers)
    log_ensemble_retriever = EnsembleRetriever(retrievers=available_retrievers, weights=equal_weighting)
    
    # 7. Semantic Chunking
    try:
        semantic_chunker = SemanticChunker(
            log_embeddings,
            breakpoint_threshold_type="percentile"
        )
        log_semantic_documents = semantic_chunker.split_documents(log_docs)
        
        log_semantic_vectorstore = Qdrant.from_documents(
            log_semantic_documents,
            log_embeddings,
            location=":memory:",
            collection_name="Log_Incident_Semantic_Chunks"
        )
        log_semantic_retriever = log_semantic_vectorstore.as_retriever(search_kwargs={"k": 5})
    except:
        print("⚠️ Semantic chunking not available, skipping")
        log_semantic_retriever = log_naive_retriever
    
    print("✅ All log retrievers created")
    
    return {
        'naive': log_naive_retriever,
        'bm25': log_bm25_retriever,
        'compression': log_compression_retriever,
        'multi_query': log_multi_query_retriever,
        'parent': log_parent_retriever,
        'ensemble': log_ensemble_retriever,
        'semantic': log_semantic_retriever,
        'vectorstore': log_vectorstore
    }