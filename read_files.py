from pathlib import Path
import re
from typing import List, Optional, Union
from uuid import uuid4

from helpers import detect_format, get_text_file_content
from models import AsciiBlock, AsciiFile, TextFileFormat


def extract_ascii_blocks(filepath: Union[str, Path], file_id: str) -> List[AsciiBlock]:
    """
    Extract a list of text blocks from an asciidoc file.

    Blank line(s) in text is used to group text lines into semantic blocks.
    """
    blocks: Optional[List[AsciiBlock]] = []
    filepath = Path(filepath)
    text_content = get_text_file_content(filepath)

    block_counter = 0
    block_text = ''

    # group text lines into semantic blocks
    for line in text_content.split('\n'):
        line_is_blank = re.fullmatch(r'\s*', line)
        
        if (
            (not block_text or re.fullmatch(r'\s*', block_text)) 
            and line_is_blank
        ):
            # add any initial blank line(s) to block
            block_text += line + '\n'
        elif not line_is_blank:
            # add text lines to block
            block_text += line + '\n'
        else:
            # stop at next blank line (but add to end of block)
            block_text += line + '\n'
            blocks.append(
                AsciiBlock(
                    index=block_counter,
                    file_id=file_id,
                    block_id=str(uuid4()),
                    original_content=block_text
                )
            )
            block_counter += 1
            block_text = ''

    block_counter += 1

    # Catch any remaining text after loop ends
    if block_text:
        blocks.append(
            AsciiBlock(
                index=block_counter,
                file_id=file_id,
                block_id=str(uuid4()),
                original_content=block_text
            )
        )

    return blocks


def read_files(filepaths: List[Union[str, Path]]) -> Optional[List[AsciiFile]]:
    """
    From a list of filepaths, read text content into
    a list of AsciiFile and AsciiBlock model instances.
    """
    filepaths = [Path(fp) for fp in filepaths]
    text_files: Optional[List[AsciiFile]] = []

    for i, path in enumerate(filepaths):
        text_file_format: TextFileFormat = detect_format(path)
        if text_file_format == 'asciidoc':
            file_id = str(uuid4())
            text_files.append(
                AsciiFile(
                   	index=i,
                    id=file_id,
                    filepath=path,
                    text_blocks=extract_ascii_blocks(path, file_id)
                )
            )
        else:
            raise ValueError(f"Unsupported file type: {path}")

    return text_files
