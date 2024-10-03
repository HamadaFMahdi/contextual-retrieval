"""
embedding_models.py

Handles embedding models for text encoding.
"""

from abc import ABC, abstractmethod
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union

class BaseEmbeddingModel(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings.

        Parameters:
        - texts (List[str]): Texts to encode.

        Returns:
        - np.ndarray: Embeddings of the texts.
        """
        pass

class SentenceTransformerModel(BaseEmbeddingModel):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu'):
        """
        Initialize the SentenceTransformer embedding model.

        Parameters:
        - model_name (str): Name of the SentenceTransformer model.
        - device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings using SentenceTransformer.

        Parameters:
        - texts (List[str]): Texts to encode.

        Returns:
        - np.ndarray: Embeddings of the texts.
        """
        return self.model.encode(texts, convert_to_numpy=True)

class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str = 'text-embedding-ada-002', api_key: str = None):
        """
        Initialize the OpenAI embedding model.

        Parameters:
        - model_name (str): Name of the OpenAI embedding model.
        - api_key (str): OpenAI API key.
        """
        import openai
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key)

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings using OpenAI's API.

        Parameters:
        - texts (List[str]): Texts to encode.

        Returns:
        - np.ndarray: Embeddings of the texts.
        """
        import openai
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(input=text, model=self.model_name)
            embeddings.append(response.data[0].embedding)
        return np.array(embeddings)

def EmbeddingModel(model_name: str = 'all-MiniLM-L6-v2', **kwargs) -> BaseEmbeddingModel:
    """
    Factory function to create an appropriate embedding model based on the model name.

    Parameters:
    - model_name (str): Name of the embedding model.
    - **kwargs: Additional keyword arguments for the specific model.

    Returns:
    - BaseEmbeddingModel: An instance of the appropriate embedding model.
    """
    if model_name.startswith('text-embedding-'):
        return OpenAIEmbeddingModel(model_name, **kwargs)
    else:
        return SentenceTransformerModel(model_name, **kwargs)