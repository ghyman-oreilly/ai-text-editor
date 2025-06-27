import logging
from pathlib import Path
import subprocess
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
