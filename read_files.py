from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from helpers import detect_format
from models import AsciiBlock, AsciiFile, TextFileFormat


def get_text_file_content():
    """
    Return content of a text file.
    """
    pass


def extract_ascii_blocks():
    """
    Extract a list of text blocks from an asciidoc file.
    """
    pass


def read_files(filepaths: List[Path]) -> Optional[List[AsciiBlock]]:
    """
    From a list of filepaths, read text content into
    a list of AsciiFile and AsciiBlock model instances.
    """
    text_files: Optional[List[AsciiFile]] = []

    for i, path in enumerate(filepaths):
        text_file_format: TextFileFormat = detect_format(path)
        if text_file_format == 'asciidoc':
            text_files.append(
                AsciiFile(
                   	index=i,
                    id=uuid4(),
                    filepath=path,
                    text_blocks=extract_ascii_blocks(path)
                )
            )
        else:
            raise ValueError(f"Unsupported file type: {path}")

    return text_files
