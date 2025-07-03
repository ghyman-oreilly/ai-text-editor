import hashlib
import json
import logging
from pathlib import Path
import subprocess
import tiktoken
from typing import Union

from models import TextFileFormat


logger = logging.getLogger(__name__)


class AsciidoctorMissingError(RuntimeError):
    """Raised when the Asciidoctor CLI is not found."""
    pass


class FileFormatError(BaseException):
    """Raised when unsupported file format is encountered."""
    pass


def check_asciidoctor_installed(raise_on_error=True) -> bool:
    """Check if the asciidoctor CLI is available on the system."""
    try:
        result = subprocess.run(
            ["asciidoctor", "-v"],
            capture_output=True,
            text=True,
            check=True
        )
        logger.debug(f"Asciidoctor version: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        message = (
            "The 'asciidoctor' CLI was not found. Please install it and ensure it's in your PATH."
        )
        logger.error(message)
        if raise_on_error:
            raise AsciidoctorMissingError(message)
        return False
    except subprocess.CalledProcessError as e:
        logger.error("Asciidoctor CLI returned an error.")
        logger.debug(f"Subprocess stderr: {e.stderr}")
        if raise_on_error:
            raise AsciidoctorMissingError("Asciidoctor CLI returned an error.") from e
        return False


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

def count_token_length(text: str, model: str = "gpt-4o") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
