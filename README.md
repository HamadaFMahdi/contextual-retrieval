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

Here's a simple example to get you started:

```python
from contextual_retrieval import ContextualRetrieval

# Initialize the retriever
retriever = ContextualRetrieval(mode='contextual_embedding')

# Index some documents
documents = [
    "Artificial Intelligence is transforming various industries.",
    "Machine Learning is a subset of AI focused on data-driven algorithms.",
    "Natural Language Processing enables computers to understand human language."
]
retriever.index_documents(documents)

# Query the system
results = retriever.query("What is AI?", top_k=2)
for doc, score in results:
    print(f"Score: {score:.4f}, Document: {doc}")
```

## Advanced Usage

For more advanced usage, including custom models and configurations, check out the [advanced example](examples/advanced_example.py).

## Components

- **EmbeddingModel**: Handles text embedding using various models.
- **ContextGenerator**: Generates context for chunks using language models.
- **BM25Retriever**: Implements BM25 retrieval functionality.
- **Reranker**: Reranks retrieved chunks using transformer-based models.
- **VectorStore**: Manages the vector database for embeddings.

## Documentation

For detailed documentation, please visit our [documentation site](https://contextual-retrieval.readthedocs.io/).

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{contextual_retrieval,
  title = {Contextual Retrieval: An Open-Source Library for Improved RAG Systems},
  author = {Your Name},
  year = {2023},
  url = {https://github.com/yourusername/contextual-retrieval}
}
```

## Support

For support, please open an issue on our [GitHub issue tracker](https://github.com/yourusername/contextual-retrieval/issues).