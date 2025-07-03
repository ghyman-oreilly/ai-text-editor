import click
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import List, Union

from ai_service import AIServiceCaller
from helpers import check_asciidoctor_installed, get_text_file_content, get_json_file_content
from models import AsciiFile, load_style_guide
from prompts import ASCII_QA_PROMPT_BASE_TEXT, COPYEDIT_PROMPT_BASE_TEXT, check_and_update_embedding_items, generate_prompt_text, generate_style_guide_text
from read_files import read_files
from write_files import write_files


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
@click.option("--disable-qa-pass", "-q", is_flag=True, help="Disable QA pass of AI service calls designed to clean up any formatting errors introduced by the model. Model tends sporadically to convert some AsciiDoc formatting to Markdown during editing pass, likely due to large and complex prompting.")
def cli(input_paths, load_data_from_json=None, disable_qa_pass=False):
    """Script for using AI to edit documents in alignment with an editorial stylesheet."""    
    chapter_filepaths = resolve_input_paths(input_paths)  
    project_dir = chapter_filepaths[0].parent
    cwd = Path(os.getcwd())
    style_guide_dir = Path(cwd / 'style_guides')
    backup_data_filepath = Path(cwd / f"{'backup_' + str(int(time.time())) + '.json'}")
    word_list_filepath = Path(style_guide_dir / 'wordlist.txt')
    word_list_w_embeddings_filepath = Path(style_guide_dir / 'wordlist_w_embeddings.npz')
    local_style_rules_filepath = Path(style_guide_dir / 'style_guide_local.json')
    local_style_rules_w_embeddings_filepath = Path(style_guide_dir / 'style_local_w_embeddings.npz')

    embedding_model = 'BAAI/bge-small-en-v1.5'

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

    ai_service_caller = AIServiceCaller(st_embedding_model=embedding_model)

    click.echo("Loading word list embeddings...")

    word_list = get_text_file_content(word_list_filepath).split('\n')

    style_guide = load_style_guide(local_style_rules_filepath, "local")

    word_list_embeddings = check_and_update_embedding_items(word_list, word_list_w_embeddings_filepath, embedding_model, ai_service_caller.generate_st_embedding)

    local_styles_list = [r["content"] for i in get_json_file_content(local_style_rules_filepath) for r in i["rules"]]

    local_style_embeddings = check_and_update_embedding_items(local_styles_list, local_style_rules_w_embeddings_filepath, embedding_model, ai_service_caller.generate_st_embedding)

    click.echo("Sending text passages to AI service for copyediting...")

    num_text_blocks = len([tb for f in all_text_files for tb in f.text_blocks])
    block_counter = 0

    # send text to AI service for block-level copyediting
    for i, text_file in enumerate(all_text_files):
        preceding_text_block = ""
        for text_block in text_file.text_blocks:
            block_counter += 1
            base_msg = f"text passage {block_counter} of {num_text_blocks} passages (file {i+1} of {len(all_text_files)})"

            if text_block.is_edited or text_block.ai_edited_content:
                click.echo(f"Skipping {base_msg} (already edited)...")
                continue

            click.echo(f"Sending {base_msg} for editing...")
            
            original_content = text_block.original_content

            deterministically_matched_style_rules = style_guide.get_matching_rule_contents(original_content, 'asciidoc')

            style_guide_text = generate_style_guide_text(
                text_passage=original_content,
                word_list_embeddings=(word_list_embeddings if word_list_embeddings else None),
                local_style_rules_embeddings=(local_style_embeddings if local_style_embeddings else None),
                embed_function=ai_service_caller.generate_st_embedding,
                other_style_rules_to_inject=(deterministically_matched_style_rules if deterministically_matched_style_rules else None)
            )

            prompt_text = generate_prompt_text(
                prompt_template=COPYEDIT_PROMPT_BASE_TEXT,
                model="gpt-4o",
                max_tokens_per_prompt=20000,
                template_kwargs={
                    "style_guide": style_guide_text,
                    "preceding_passage": preceding_text_block,
                    "format_type": 'asciidoc',
                    "passage_to_be_edited": original_content,
                }
            )

            prompt = ai_service_caller.create_prompt_object(prompt_text)
            edited_text = ai_service_caller.call_ai_service(prompt)

            if edited_text:
                text_block.ai_edited_content = edited_text
                text_block.is_edited = True

            # set preceding block for next block
            preceding_text_block = edited_text or text_block.original_content

            write_backup_to_json_file(all_text_files, backup_data_filepath)
            click.echo(f"\nText files data backed up to {backup_data_filepath}...\n")

    click.echo("Sending edited text to AI service for QA...")

    # if all edited, send original text and edited to AI service for QA
    if (
        all(b.is_edited for f in all_text_files for b in f.text_blocks)
        and not disable_qa_pass
    ):
        block_counter = 0
        no_issue_str = 'NO_ISSUE'
        for i, text_file in enumerate(all_text_files):
            for text_block in text_file.text_blocks:
                block_counter += 1
                base_msg = f"text passage {block_counter} of {num_text_blocks} passages (file {i+1} of {len(all_text_files)})"

                if text_block.is_qaed:
                    click.echo(f"Skipping {base_msg} (already QAed)...")
                    continue

                if text_block.original_content == text_block.ai_edited_content:
                    click.echo(f"Skipping {base_msg} (no changes in edited text)...")
                    continue

                click.echo(f"Sending {base_msg} for QA...")
                
                prompt_text = generate_prompt_text(
                    prompt_template=ASCII_QA_PROMPT_BASE_TEXT,
                    model="gpt-4o",
                    max_tokens_per_prompt=10000,
                    template_kwargs={
                        'no_issue_str': no_issue_str,
                        'original_text': text_block.original_content,
                        'edited_text': text_block.ai_edited_content
                    }
                )

                prompt = ai_service_caller.create_prompt_object(prompt_text)
                response = ai_service_caller.call_ai_service(prompt)

                if response:
                    if response.strip().lower() != no_issue_str.lower():
                        text_block.ai_qaed_content = response
                    text_block.is_qaed = True

                write_backup_to_json_file(all_text_files, backup_data_filepath)
                click.echo(f"\nText files data backed up to {backup_data_filepath}...\n")     

    # if all processed and QAed, save edited text to files
    if all(b.is_edited and (disable_qa_pass or b.is_qaed) for f in all_text_files for b in f.text_blocks):
        click.echo("Writing edited text to source files...")
        write_files(all_text_files)

    click.echo("Script complete.")


if __name__ == '__main__':
    cli()