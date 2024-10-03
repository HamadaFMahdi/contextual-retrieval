"""
advanced_example.py

An advanced usage example with custom models and settings.
"""

import torch
from contextual_retrieval import ContextualRetrieval, EmbeddingModel, ContextGenerator, Reranker, BM25Retriever, set_api_key
from contextual_retrieval.utils import truncate_text, count_tokens

# Set the API key (you can also set it as an environment variable)
set_api_key("your-openai-api-key-here")

# Custom embedding model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model = EmbeddingModel(model_name='sentence-transformers/all-mpnet-base-v2', device=device)

# Custom context generator with a custom prompt
custom_prompt = ("Please provide a brief summary to place this excerpt within the larger context of the document. "
                 "Only provide the summary and nothing else.")
context_generator = ContextGenerator(model_name='gpt-4o-mini', prompt=custom_prompt)

# Custom reranker
reranker = Reranker(model_name='cross-encoder/ms-marco-MiniLM-L-12-v2', device=device)

# Custom BM25 retriever
bm25_retriever = BM25Retriever()

# Initialize the retriever with custom components and 'rerank' mode
retriever = ContextualRetrieval(
    mode='rerank',
    embedding_model=embedding_model,
    context_generator=context_generator,
    bm25_retriever=bm25_retriever,
    reranker=reranker,
    chunk_size=256,
    chunk_overlap=50,
)

# Sample documents
documents = [
    "Deep learning has revolutionized the field of computer vision. Convolutional neural networks are a core technology in image recognition and object detection. These networks can automatically learn hierarchical features from raw pixel data.",
    "Natural language processing allows computers to understand human language. It has applications in machine translation, sentiment analysis, and chatbots. Recent advances in transformer models have significantly improved the performance of NLP tasks.",
    "Reinforcement learning enables AI to learn through interaction with the environment. It's used in robotics, game playing, and autonomous systems. Q-learning and policy gradient methods are popular approaches in reinforcement learning.",
    "The ethical implications of AI are a growing concern. Issues such as bias in AI systems, privacy concerns, and the potential for job displacement need to be addressed. Responsible AI development is crucial for the technology's long-term success and acceptance.",
]

# Index the documents
print("Indexing documents...")
retriever.index_documents(documents)
print(f"Indexed {retriever.get_indexed_document_count()} documents")

# User queries
queries = [
    "How does AI understand human language?",
    "What are the ethical concerns surrounding AI?",
    "Explain the use of neural networks in computer vision.",
]

# Process each query
for query in queries:
    print(f"\nQuery: {query}")
    print(f"Query token count: {count_tokens(query)}")
    
    results = retriever.query(query, top_k=2)

    print("\nTop Results:")
    for i, (chunk, score) in enumerate(results):
        print(f"{i+1}. (Score: {score:.4f})")
        print(f"   {truncate_text(chunk, 100)}")
        print(f"   Token count: {count_tokens(chunk)}")

# Save the vector store
print("\nSaving the vector store...")
retriever.vector_store.save("vector_store.faiss", "documents.pkl")

# Clear the index
print("Clearing the index...")
retriever.clear_index()
print(f"Remaining indexed documents: {retriever.get_indexed_document_count()}")

# Load the saved vector store
print("Loading the saved vector store...")
new_retriever = ContextualRetrieval(
    mode='contextual_embedding',
    embedding_model=embedding_model,
    vector_store=retriever.vector_store
)
new_retriever.vector_store.load("vector_store.faiss", "documents.pkl")
print(f"Loaded {new_retriever.get_indexed_document_count()} documents")

# Try a query with the loaded vector store
query = "What is reinforcement learning?"
results = new_retriever.query(query, top_k=1)
print(f"\nQuery: {query}")
print("Top Result:")
print(f"(Score: {results[0][1]:.4f}) {results[0][0]}")