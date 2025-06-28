import click
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import List, Union

from helpers import check_asciidoctor_installed
from models import AsciiFile
from read_files import read_files


# config root logger
logging.basicConfig(
    level=logging.INFO, # Allow INFO and above to be printed
    stream=sys.stdout   # Print to console
)
# elevate min logging level for noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

SUPPORTED_FILE_EXT = [".adoc", ".asciidoc"]


def read_json_file_list(filepath: Path) -> list[Path]:
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
        if abs_path.exists() and abs_path.is_file() and abs_path.suffix.lower() in SUPPORTED_FILE_EXT:
            filepaths.append(abs_path)

    if not filepaths:
        raise ValueError("No files found in the JSON `file` list.")

    return filepaths


def resolve_input_paths(inputs: list) -> list[Path]:
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
                f for f in input_path.glob("*") if f.exists() and f.is_file() and f.suffix.lower() in SUPPORTED_FILE_EXT
            )

        # handle filepaths input
        elif input_path.is_file():
            if input_path.suffix.lower() in SUPPORTED_FILE_EXT:
                resolved_files.append(input_path)
            else:
                raise ValueError(
                    f"If passing a list of filepaths, all files must be of types: {', '.join(SUPPORTED_FILE_EXT)}"
                    )
        
        else:
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

    return resolved_files


def sort_chapter_files_by_json_file_list(chapter_filepaths: list[Path], json_file_list_filepaths: Union[list[Path], None] = None) -> list[Path]:
    """
    Sort chapter_filepaths according to relative paths in JSON file list.
    Files not in json_file_list_filepaths will come after, sorted alphabetically.
    If json_file_list_filepaths is None or empty, sort chapter_filepaths alphabetically.
    """

    if not json_file_list_filepaths:
        # Sort alphabetically by filename (or full path)
        return sorted(chapter_filepaths, key=lambda p: p.name.lower())

    # Create a lookup map from filename (or relative path) to order
    file_list_order = {Path(p).name: i for i, p in enumerate(json_file_list_filepaths)}

    def sort_key(path: Path):
        # If path.name is in atlas, return its order. Otherwise, sort after all atlas items.
        return (file_list_order.get(path.name, float('inf')), path.name.lower())

    return sorted(chapter_filepaths, key=sort_key)

def write_backup_to_json_file(input_data: list[AsciiFile], output_filepath: Union[str, Path]):
    """ 
    Serialize AsciiFile model data and save in JSON file.
    """
    with open(str(output_filepath), 'w') as f:
        json.dump([i.model_dump(mode="json") for i in input_data], f)


def read_backup_from_json_file(input_filepath: Union[str, Path]) -> List[AsciiFile]:
    """ 
    Read JSON file and and validate data as AsciiFile model data.
    """
    with open(str(input_filepath), 'r') as f:
        file_data = json.load(f)
        return [AsciiFile.model_validate(fd) for fd in file_data]
    

@click.command(help="""
Provide one of the following:
(1) path to a directory containing Asciidoc files,
(2) space-delimited paths to such files,
(3) path to a JSON file with a 'files' list of such files.
""")
@click.version_option(version='1.0.0')
@click.argument("input_paths", nargs=-1)
@click.option("--load-data-from-json", "-l", default=None, help="Provide the path to an optional JSON file of data backed up from a previous session. Useful for continuing your progress after a session is interrupted, without having to send all data back to the AI service. NOTE: Do not use this option if you've made changes in the repo since the backup file was produced, as it may overwrite your changes.")
def cli(input_paths, load_data_from_json=None):
    """Script for using AI to edit documents in alignment with an editorial stylesheet."""    
    chapter_filepaths = resolve_input_paths(input_paths)  
    project_dir = chapter_filepaths[0].parent
    cwd = Path(os.getcwd())
    backup_data_filepath = Path(cwd / f"{'backup_' + str(int(time.time())) + '.json'}")

    # look for JSON filelist for sorting chapter files
    speculative_atlas_json_filepath = Path(project_dir / "atlas.json")
    atlas_filepaths = None

    if speculative_atlas_json_filepath.exists:
        atlas_filepaths = read_json_file_list(speculative_atlas_json_filepath)

    # validate load file
    if json_input_path := load_data_from_json:
        json_input_path = Path(json_input_path)
        if not json_input_path.is_file() or json_input_path.suffix.lower() != '.json':
            click.echo(f"Path provided with --load-data-from-json option must point to a valid JSON file. Exiting.")
            click.Abort()

    # sort by filelist or alpha
    sorted_filepaths = sort_chapter_files_by_json_file_list(chapter_filepaths, atlas_filepaths)

    filelist_str = '\n'.join([str(f) for f in sorted_filepaths])

    click.echo(f"Files to be processed include:\n{filelist_str}")
    
    if not click.prompt("Do you wish to continue? (y/n)").strip().lower() in ['y', 'yes']:
        click.echo("Exiting.")
        sys.exit(0)

    if json_input_path:
        # load from JSON file
        print("Loading repo data from JSON file...")
        all_text_files: List[AsciiFile] = read_backup_from_json_file(json_input_path)
    else:
        # read repo files
        if any(f.name.lower().endswith(('.asciidoc', '.adoc')) for f in sorted_filepaths):
            # check for asciidoctor
            check_asciidoctor_installed()

        click.echo("\nProgress: Script is running. This may take up to several minutes to complete. Please wait...\n")

        files_to_skip = [] # for use with potential future HTML use case

        # collect text file data
        click.echo("\nExtracting data from text files...\n")
        all_text_files: List[AsciiFile] = read_files(sorted_filepaths)

        write_backup_to_json_file(all_text_files, backup_data_filepath)
        click.echo(f"\nText files data backed up to {backup_data_filepath}...\n")

    click.echo("Script complete.")


if __name__ == '__main__':
    cli()