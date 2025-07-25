import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Union

from models import TextFileFormat


logger = logging.getLogger(__name__)


class FileFormatError(BaseException):
    """Raised when unsupported file format is encountered."""
    pass


def detect_format(path: Path) -> TextFileFormat:
    """
    Return a TextFileFormat string
    """
    if path.suffix.lower() in (".asciidoc", ".adoc"):
        return "asciidoc"
    else:
        raise FileFormatError(f"Unsupported file format: {path.suffix.lower()}")


def get_text_file_content(filepath: Union[str, Path]) -> str:
    """
    Return content of a text file.
    """
    with open(str(filepath), 'r', encoding='utf-8') as f:
        return f.read()

def write_text_to_file(filepath: Union[str, Path], text_content: str) -> str:
    """
    Write text to file.
    """
    with open(str(filepath), 'w', encoding='utf-8') as f:
        f.write(text_content)

def write_json_to_file(filepath: Union[str, Path], data):
    with open(str(filepath), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def get_json_file_content(filepath: Union[str, Path]):
    with open(str(filepath), "r", encoding="utf-8") as f:
        return json.load(f)

def count_token_length(text: str, model: str = "gpt-4o", encoding: str = "cl100k_base") -> int:
    """
    Count token length of input text.

    cl100k_base encoding is used by gpt-4o, 3o, gpt-4.1, etc.
    """
    import tiktoken
    try:
        enc = tiktoken.encoding_for_model(model)
    except:
        enc = tiktoken.get_encoding(encoding)
    return len(enc.encode(text))

def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def clean_response(response: str, original_text: str = ""):
    """
    Remove code fences from model response
    """
    cleaned = response

    # Remove leading code fence if the original did not start with one
    if re.match(r'```(?:[a-zA-Z0-9]+)?\s*', response) and not original_text.strip().startswith("```"):
        cleaned = re.sub(r'^```(?:[a-zA-Z0-9]+)?\s*', '', cleaned)

    # Remove trailing code fence if the original did not end with one
    if re.search(r'```$', cleaned.strip()) and not original_text.strip().endswith("```"):
        cleaned = re.sub(r'\s*```$', '', cleaned)

    return cleaned
