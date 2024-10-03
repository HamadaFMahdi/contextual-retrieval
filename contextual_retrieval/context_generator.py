"""
context_generator.py

Generates context for chunks using language models.
"""

from abc import ABC, abstractmethod
import openai
from typing import List, Optional

class BaseContextGenerator(ABC):
    @abstractmethod
    def generate_context(self, chunks: List[str], documents: Optional[List[str]] = None) -> List[str]:
        """
        Generate context for each chunk.

        Parameters:
        - chunks (List[str]): Text chunks.
        - documents (Optional[List[str]]): The original documents corresponding to each chunk.

        Returns:
        - List[str]: List of contextualized chunks.
        """
        pass

class OpenAIContextGenerator(BaseContextGenerator):
    def __init__(self, model_name: str = 'gpt-3.5-turbo', api_key: Optional[str] = None, prompt: Optional[str] = None):
        """
        Initialize the OpenAI context generator.

        Parameters:
        - model_name (str): Name of the OpenAI model to use.
        - api_key (Optional[str]): API key for OpenAI.
        - prompt (Optional[str]): Custom prompt for generating context.
        """
        self.model_name = model_name
        self.prompt = prompt or self.default_prompt()

        if api_key:
            openai.api_key = api_key
        elif not openai.api_key:
            raise ValueError("OpenAI API key must be provided either during initialization or set as an environment variable.")

    @staticmethod
    def default_prompt() -> str:
        return ("Please give a short succinct context to situate this chunk within the overall document "
                "for the purposes of improving search retrieval of the chunk. Answer only with the succinct "
                "context and nothing else.")

    def generate_context(self, chunks: List[str], documents: List[str]) -> List[str]:
        """
        Generate context for each chunk using the parent document.

        Parameters:
        - chunks (List[str]): Text chunks.
        - documents (List[str]): Parent documents for each chunk.

        Returns:
        - List[str]: List of contextualized chunks.
        """
        contextualized_chunks = []
        for i, chunk in enumerate(chunks):
            document = documents[i]
            try:
                context = self._call_model(document, chunk)
                contextualized_chunk = f"{context} {chunk}"
                contextualized_chunks.append(contextualized_chunk)
            except Exception as e:
                print(f"Error generating context for chunk {i}: {str(e)}")
                contextualized_chunks.append(chunk)  # Use original chunk if context generation fails
        return contextualized_chunks

    def _call_model(self, document: str, chunk: str) -> str:
        """
        Call the OpenAI model to generate context.

        Parameters:
        - document (str): The full document.
        - chunk (str): The chunk to generate context for.

        Returns:
        - str: The generated context.
        """
        prompt = (
            f"<document>\n{document}\n</document>\n"
            f"Here is the chunk we want to situate within the overall document:\n"
            f"<chunk>\n{chunk}\n</chunk>\n"
            f"{self.prompt}"
        )

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                n=1,
                temperature=0.7,
            )
            context = response['choices'][0]['message']['content'].strip()
            return context
        except openai.error.OpenAIError as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")

def ContextGenerator(model_name: str = 'gpt-3.5-turbo', **kwargs) -> BaseContextGenerator:
    """
    Factory function to create an appropriate context generator based on the model name.

    Parameters:
    - model_name (str): Name of the context generation model.
    - **kwargs: Additional keyword arguments for the specific model.

    Returns:
    - BaseContextGenerator: An instance of the appropriate context generator.
    """
    if 'gpt' in model_name:
        return OpenAIContextGenerator(model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported model: {model_name}")