"""
utils.py

Utility functions for text processing and document handling.
"""

from typing import List, Optional
from nltk.tokenize import word_tokenize, sent_tokenize


def chunk_documents(documents: List[str], chunk_size: int = 512, chunk_overlap: int = 50) -> List[str]:
    """
    Chunk the documents into smaller pieces.

    Parameters:
    - documents (List[str]): The list of documents to chunk.
    - chunk_size (int): The size of each chunk in tokens.
    - chunk_overlap (int): The number of tokens to overlap between chunks.

    Returns:
    - List[Tuple[str, str]]: List of tuples (chunk, parent_document).
    """
    chunks = []
    for doc in documents:
        doc_chunks = chunk_text(doc, chunk_size, chunk_overlap)
        chunks.extend([(chunk, doc) for chunk in doc_chunks])
    return chunks

def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> List[str]:
    """
    Chunk a single text into smaller pieces.

    Parameters:
    - text (str): The text to chunk.
    - chunk_size (int): The size of each chunk in tokens.
    - chunk_overlap (int): The number of tokens to overlap between chunks.

    Returns:
    - List[str]: List of chunks.
    """
    tokens = word_tokenize(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk = ' '.join(chunk_tokens)
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks

def truncate_text(text: str, max_length: int, truncation_marker: str = "...") -> str:
    """
    Truncate text to a maximum length, ensuring it doesn't cut off in the middle of a word.

    Parameters:
    - text (str): The text to truncate.
    - max_length (int): The maximum length of the truncated text.
    - truncation_marker (str): The string to append to truncated text.

    Returns:
    - str: Truncated text.
    """
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    if last_space != -1:
        truncated = truncated[:last_space]
    
    return truncated.rstrip() + truncation_marker

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.

    Parameters:
    - text (str): The text to split.

    Returns:
    - List[str]: List of sentences.
    """
    return sent_tokenize(text)

def remove_special_characters(text: str, keep_chars: Optional[str] = None) -> str:
    """
    Remove special characters from text, optionally keeping specified characters.

    Parameters:
    - text (str): The text to process.
    - keep_chars (Optional[str]): String of characters to keep.

    Returns:
    - str: Processed text.
    """
    import re
    if keep_chars:
        pattern = f"[^a-zA-Z0-9\s{re.escape(keep_chars)}]"
    else:
        pattern = r"[^a-zA-Z0-9\s]"
    return re.sub(pattern, "", text)

def count_tokens(text: str) -> int:
    """
    Count the number of tokens in the text.

    Parameters:
    - text (str): The text to count tokens from.

    Returns:
    - int: Number of tokens.
    """
    return len(word_tokenize(text))