"""
simple_example.py

A simple usage example of the Contextual Retrieval library.
"""

from contextual_retrieval import ContextualRetrieval
from contextual_retrieval.embedding_models import EmbeddingModel
from contextual_retrieval.context_generator import ContextGenerator
from contextual_retrieval.reranker import Reranker

# Sample documents
documents = [
    "This is a document about machine learning and artificial intelligence. It discusses various algorithms and techniques.",
    "Here we discuss the applications of AI in various industries, including healthcare, finance, and transportation.",
    "The future of technology is shaped by advances in AI and ML. Innovations are happening rapidly.",
    "Natural Language Processing (NLP) is a subfield of AI that focuses on the interaction between computers and humans using natural language.",
    "Computer vision is another important area of AI, dealing with how computers can understand and process visual information from the world.",
]

# Initialize the retriever with custom components
embedding_model = EmbeddingModel('all-MiniLM-L6-v2')
context_generator = ContextGenerator(model_name='gpt-3.5-turbo')
reranker = Reranker()

retriever = ContextualRetrieval(
    mode='rerank',
    embedding_model=embedding_model,
    context_generator=context_generator,
    reranker=reranker
)

# Index the documents
print("Indexing documents...")
retriever.index_documents(documents)
print(f"Indexed {retriever.get_indexed_document_count()} documents")

# User queries
queries = [
    "What are the applications of artificial intelligence?",
    "How is AI shaping the future of technology?",
    "What is Natural Language Processing?",
]

# Retrieve relevant chunks for each query
for query in queries:
    print(f"\nQuery: {query}")
    results = retriever.query(query, top_k=2)

    print("Top Results:")
    for i, (chunk, score) in enumerate(results):
        print(f"{i+1}. (Score: {score:.4f}) {chunk}")

# Clear the index
print("\nClearing the index...")
retriever.clear_index()
print(f"Remaining indexed documents: {retriever.get_indexed_document_count()}")