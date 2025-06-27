import click
import json
import logging
from pathlib import Path
import sys


# config root logger
logging.basicConfig(
    level=logging.INFO, # Allow INFO and above to be printed
    stream=sys.stdout   # Print to console
)
# elevate min logging level for noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def read_json_file_list(
        filepath: Path,
        permitted_filetypes: list[str] = [".html", ".adoc", ".asciidoc"]
        ) -> list[Path]:
    """
    Get list of files from a `file` list in a JSON file.

    Expects `file` list to contain strings representing relative paths
    to files in the same directory as the JSON file.
    """
    filepath = Path(filepath) 
    project_dir = filepath.parent

    filepaths = []

    try:
        json_data = json.loads(filepath.read_text(encoding='utf-8'))
    except Exception as e:
        raise Exception(f"Failed to read or parse JSON file: {e}")

    for rel_path in json_data.get("files", []):
        abs_path = (project_dir / rel_path).resolve()
        if abs_path.exists() and abs_path.is_file() and abs_path.suffix.lower() in permitted_filetypes:
            filepaths.append(abs_path)

    if not filepaths:
        raise ValueError("No files found in the JSON `file` list.")

    return filepaths


def resolve_input_paths(
        inputs: list, 
        permitted_filetypes: list[str] = [".adoc", ".asciidoc"]
    ) -> list[Path]:
    """
    Takes in a list of input paths and resolves them into a list of Path objects.
    Supports:
    - A folder path containing files
    - A path to a .json file with a `files` key
    - One or more individual file paths
    """
    resolved_files = []

    json_filepath = next(
        (Path(i) for i in inputs if Path(i).is_file() and Path(i).suffix.lower() == '.json'),
        ''
    )

    # handle JSON input
    if json_filepath:
        resolved_files = read_json_file_list(Path(json_filepath))
        return resolved_files

    for input_str in inputs:
        input_path = Path(input_str)

        # handle directory input
        if input_path.is_dir():
            resolved_files.extend(
                f for f in input_path.glob("*") if f.exists() and f.is_file() and f.suffix.lower() in permitted_filetypes
            )

        # handle filepaths input
        elif input_path.is_file():
            if input_path.suffix.lower() in permitted_filetypes:
                resolved_files.append(input_path)
            else:
                raise ValueError(
                    f"If passing a list of filepaths, all files must be of types: {', '.join(permitted_filetypes)}"
                    )
        
        else:
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

    return resolved_files


@click.command(help="""
Provide one of the following:
(1) path to a directory containing Asciidoc files,
(2) space-delimited paths to such files,
(3) path to a JSON file with a 'files' list of such files.
""")
@click.version_option(version='1.0.0')
@click.argument("input_paths", nargs=-1)
def cli(input_paths):
    """Script for using AI to edit documents in alignment with an editorial stylesheet."""
    pass
    

if __name__ == '__main__':
    cli()