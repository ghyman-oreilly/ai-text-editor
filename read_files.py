from pathlib import Path
import re
from typing import List, Optional, Union
from uuid import uuid4

from helpers import count_token_length, detect_format, get_text_file_content
from models import AsciiBlock, AsciiFile, TextFileFormat


def is_attribute_line(line: str) -> bool:
    return (
        line.strip().startswith(":")  # document attribute like :chapter: 1
        or re.match(r'^\[\[.*\]\]', line.strip())  # ID ref like [[chapter_00]]
        or re.match(r'^\[[^\]]+\]', line.strip())  # Role, options like [.lead], [role="..."]
    )

def is_section_heading(line: str) -> bool:
    # True if it’s a heading (min 1 =, space required), but not a block delimiter (like ====)
    return bool(re.match(r'^={1,6} [^\n]+$', line.strip()))

def split_into_sections(text: str) -> List[str]:
    """
    Splits AsciiDoc content into logical sections by headings,
    while preserving:
    - Document-level attributes
    - Block-level metadata above headings (e.g., [[id]], [.role])
    - Optional preamble, which is merged with the first section
    """
    lines = text.splitlines(keepends=True)
    sections: List[List[str]] = []
    current_section: List[str] = []
    pending_attrs: List[str] = []

    for line in lines:
        stripped = line.strip()

        if is_attribute_line(stripped):
            # Buffer attribute lines (could be doc-level or heading-bound)
            pending_attrs.append(line)
            continue

        if is_section_heading(stripped):
            # Flush current section
            if current_section:
                sections.append(current_section)

            # Start new section with any pending attributes + heading
            current_section = pending_attrs + [line]
            pending_attrs = []
        else:
            # Content line: attach any pending attributes to current section
            if pending_attrs:
                current_section.extend(pending_attrs)
                pending_attrs = []
            current_section.append(line)

    if current_section:
        sections.append(current_section)

    # Convert to strings and clean up
    sections = [''.join(sec).strip() for sec in sections if ''.join(sec).strip()]

    # Merge preamble into first heading section
    if len(sections) > 1 and not re.search(r'^={1,6} ', sections[0], re.MULTILINE):
        sections[1] = sections[0] + '\n\n' + sections[1]
        sections = sections[1:]

    return sections


def is_list_item(line: str) -> bool:
    return bool(re.match(r'^\s*([-*+]|\d+[\.\)])\s+', line))


def is_admonition_start(line: str) -> bool:
    return bool(re.match(r'^\[(NOTE|TIP|IMPORTANT|WARNING|CAUTION|TODO)\]', line))


def is_delimited_block_start(line: str) -> bool:
    return line.strip() in ("----", "++++", "____", "****", "|===")


def is_block_macro(line: str) -> bool:
    return bool(
        re.match(r'^(image|include)::', line)
    )


def group_snippets(lines: List[str]) -> List[str]:
    """
    Groups lines into block-aware snippets. Treats code blocks, lists, admonitions, etc. as atomic.
    """
    snippets: List[str] = []
    buffer: List[str] = []
    inside_block = False
    block_delimiter = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Enter or exit a delimited block scope (e.g. ----, ++++, etc.)
        if is_delimited_block_start(stripped):
            if not inside_block:
                inside_block = True
                block_delimiter = stripped
            elif stripped == block_delimiter:
                inside_block = False
                block_delimiter = None

        buffer.append(line)

        # Flush buffer at a blank line ONLY if not inside a block
        if stripped == "" and not inside_block:
            if buffer:
                snippets.append("\n".join(buffer).strip())
                buffer = []

    # Final flush
    if buffer:
        snippets.append("\n".join(buffer).strip())

    return [s for s in snippets if s.strip()]


def extract_ascii_blocks(
    filepath: Union[str, Path],
    file_id: str,
    model: str = "gpt-4o",
    max_tokens_per_block: int = 1500
) -> List[AsciiBlock]:
    """
    Parses an AsciiDoc file and emits token-bounded blocks that:
    - Respect section boundaries
    - Maintain block-level integrity for lists, code, admonitions
    - Group semantic snippets into <= max_tokens_per_block AsciiBlocks
    """
    filepath = Path(filepath)
    text = get_text_file_content(filepath)
    sections = split_into_sections(text)

    all_blocks: List[AsciiBlock] = []
    block_index = 0

    for section in sections:
        lines = section.splitlines()
        snippets = group_snippets(lines)

        buffer = ""
        buffer_tokens = 0

        for snippet in snippets:
            snippet = snippet
            if not snippet:
                continue

            snippet_tokens = count_token_length(snippet + "\n\n", model=model)

            if snippet_tokens > max_tokens_per_block:
                # Flush buffer first
                if buffer:
                    all_blocks.append(AsciiBlock(
                        index=block_index,
                        file_id=file_id,
                        block_id=str(uuid4()),
                        original_content=buffer
                    ))
                    block_index += 1
                    buffer = ""
                    buffer_tokens = 0

                all_blocks.append(AsciiBlock(
                    index=block_index,
                    file_id=file_id,
                    block_id=str(uuid4()),
                    original_content=snippet  # Let long snippet through unmodified
                ))
                block_index += 1
                continue

            if buffer_tokens + snippet_tokens > max_tokens_per_block:
                # Flush current buffer as a block
                all_blocks.append(AsciiBlock(
                    index=block_index,
                    file_id=file_id,
                    block_id=str(uuid4()),
                    original_content=buffer
                ))
                block_index += 1
                buffer = snippet + "\n\n"
                buffer_tokens = snippet_tokens
            else:
                buffer += snippet + "\n\n"
                buffer_tokens += snippet_tokens

        # Final flush per section
        if buffer:
            all_blocks.append(AsciiBlock(
                index=block_index,
                file_id=file_id,
                block_id=str(uuid4()),
                original_content=buffer
            ))
            block_index += 1

    return all_blocks


def read_files(
        filepaths: List[Union[str, Path]],
        model: str
        ) -> Optional[List[AsciiFile]]:
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
                    text_blocks=extract_ascii_blocks(path, file_id, model)
                )
            )
        else:
            raise ValueError(f"Unsupported file type: {path}")

    return text_files
