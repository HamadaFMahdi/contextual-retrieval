# Contextual Retrieval

An open-source Python library for Contextual Retrieval, designed to significantly improve the retrieval step in Retrieval-Augmented Generation (RAG) systems.

## Features

- **Easy to Use**: Get started with just a few lines of code.
- **Modular Design**: Choose between different retrieval modes:
  - Contextual Embedding
  - Contextual Embedding + Contextual BM25
  - Reranked Contextual Embedding + Contextual BM25
- **Model Agnostic**: Use your preferred models for context generation, embeddings, and reranking.
- **Customizable**: Override prompts and configurations to suit your use case.
- **Beginner Friendly**: Sensible defaults make it easy for beginners.
- **Efficient**: Utilizes FAISS for fast similarity search.
- **Flexible**: Supports both CPU and GPU acceleration.

## Installation

Install the library using pip:

```bash
pip install contextual-retrieval
```

## Quick Start

Here's a simple example to get you started with the full power of Contextual Retrieval:

```python
from contextual_retrieval import ContextualRetrieval

# Initialize the retriever with the full mode
retriever = ContextualRetrieval(mode='rerank')

# Index some documents
documents = [
    "Artificial Intelligence is transforming various industries.",
    "Machine Learning is a subset of AI focused on data-driven algorithms.",
    "Natural Language Processing enables computers to understand human language.",
    "Deep Learning models, like neural networks, are inspired by the human brain.",
    "Computer Vision allows machines to interpret and make decisions based on visual data."
]
retriever.index_documents(documents)

# Query the system
query = "What are the main areas of AI?"
results = retriever.query(query, top_k=3)

print(f"Query: {query}\n")
print("Top Results:")
for i, (doc, score) in enumerate(results, 1):
    print(f"{i}. (Score: {score:.4f}) {doc}")
```

This example demonstrates how to use the full Reranked Contextual Embedding + Contextual BM25 mode with just one line of initialization. The system will automatically generate context for chunks, use both embedding and BM25 for retrieval, and apply reranking to provide the most relevant results.

## Learn More

For more information about the Contextual Retrieval technique and its benefits, check out the original article by Anthropic: [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)

## Advanced Usage

For more advanced usage, including custom models and configurations, check out the [advanced example](examples/advanced_example.py).

## Components

- **EmbeddingModel**: Handles text embedding using various models.
- **ContextGenerator**: Generates context for chunks using language models.
- **BM25Retriever**: Implements BM25 retrieval functionality.
- **Reranker**: Reranks retrieved chunks using transformer-based models.
- **VectorStore**: Manages the vector database for embeddings.

## Contributing

We welcome contributions! 

## License

This project is licensed under the MIT License.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{contextual_retrieval,
  title = {Contextual Retrieval: An Open-Source Library for Improved RAG Systems},
  author = {Hamada Fadil Mahdi},
  year = {2024},
  url = {https://github.com/HamadaFMahdi/contextual-retrieval}
}
```