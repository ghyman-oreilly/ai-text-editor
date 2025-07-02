import hashlib
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Union

from ai_service import AIServiceCaller
from helpers import count_token_length, get_text_file_content, get_json_file_content, write_json_to_file


class MaxTokensExceeded(Exception):
	"""Raise when prompt exceeds max token limit"""
	pass


logger = logging.getLogger(__name__)


COPYEDIT_PROMPT_BASE_TEXT = """
You are an expert copyeditor. Edit the text below for grammar, clarity, and style, using the provided style guide.

[BEGIN STYLE GUIDE]
{style_guide}
[END STYLE GUIDE]

Use the preceding passage (if available) to ensure continuity, e.g., resolve pronouns, maintain tone, or complete broken lists.

[BEGIN PRECEDING PASSAGE]
{preceding_passage}
[END PRECEDING PASSAGE]

Now edit the following passage:

[BEGIN PASSAGE TO EDIT]
{passage_to_be_edited}
[END PASSAGE TO EDIT]

Output only the edited version of the passage — no explanations, commentary, or notes.

Preserve all formatting and markup consistent with {format_type}. Do not assume, infer, or substitute any other markup style under any circumstances. Do NOT use Markdown formatting; only {format_type}. Leave all formatting exactly as it appears in the input. Your only task is to edit the text for grammar, clarity, and style, in accordance with the style guide.

Do NOT enclose your response in triple backticks (```) or add a language tag.
"""

ASCII_QA_PROMPT_BASE_TEXT = """
You are an expert in AsciiDoc formatting.

Compare the original and edited text shown below. The edited text must not introduce any formatting changes that are inconsistent with AsciiDoc syntax.

If formatting errors have been introduced (e.g., AsciiDoc links changed to Markdown links, periods removed from image captions, AsciiDoc headings changed to Markdown headings), fix them — but only those formatting issues.

Do not modify grammar or writing style.

If no formatting issues have been introduced, respond exactly with this word on a single line:

{no_issue_str}

Otherwise, return the entire corrected passage, and nothing else. Do not return only a partial edit or individual sentence. Always return the full edited text with formatting corrected if needed.

[BEGIN ORIGINAL]

{original_text}

[END ORIGINAL]

[BEGIN EDITED]

{edited_text}

[END EDITED]

Return either exactly {no_issue_str} or the fully corrected version of the edited passage. Do not explain, justify, or add instructions.
"""


def generate_prompt_text(
    prompt_template: str,
    model: str,
    max_tokens_per_prompt: int,
    template_kwargs: Dict[str, str],
) -> str:
    """
    General-purpose prompt builder using any string template and dynamic field values.

    Args:
        prompt_template: The prompt template string (e.g., a f-string or str.format-compatible template).
        model: The model name (used for token-length estimation).
        max_tokens_per_prompt: Max allowed token count for the final prompt.
        template_kwargs: Dictionary of keyword substitutions that match the template's placeholders.

    Returns:
        Fully formatted prompt string, ready for LLM input.

    Raises:
        MaxTokensExceeded: If resulting token count exceeds model-specific limits.
    """
    try:
        prompt_text = prompt_template.format(**template_kwargs)
    except KeyError as e:
        raise ValueError(f"Missing a required template field: {e}")

    tokens = count_token_length(prompt_text, model)

    if tokens > max_tokens_per_prompt:
        raise MaxTokensExceeded(
            f"Prompt has {tokens} tokens, which exceeds model limit of {max_tokens_per_prompt}"
        )

    return prompt_text


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_embeddings_dict_from_list(
        input_text_list: List[str],
        ai_service_caller: AIServiceCaller
    ) -> List[Dict[str, Any]]:
    """
    For a list of text strings, generate a list of dictionaries:
        "term": the string
        "embedding: the embedding
        "text_hash": hash for checking content changes
    """
    results = []
    for i, input_text in enumerate(input_text_list):
        logger.info(f"Generating embedding for {i+1} of {len(input_text_list)} word list items...")
        embedding = ai_service_caller.generate_embedding(input_text=input_text)
        result = {
            "term": input_text,
            "embedding": embedding,
            "text_hash": compute_hash(input_text)
        }
        results.append(result)
    return results


def check_and_update_wordlist_embeddings(
        word_list_filepath: Union[Path, str],
        word_list_w_embeddings_filepath: Union[Path, str],
        ai_service_caller: AIServiceCaller,    
    ):
    """
    Check for persisted wordlist embeddings. Load or create
    new if not available or changed. 

    Return list of dicts:
        "term": wordlist item (string),
        "embedding": embedding,
        "text_hash": hash for checking content changes
    """
    word_list_filepath = Path(word_list_filepath)
    word_list_w_embeddings_filepath = Path(word_list_w_embeddings_filepath)

    if (
        not word_list_filepath.exists 
        or not word_list_filepath.is_file
        or not word_list_filepath.suffix.lower() == '.txt'
        ):
        raise ValueError(f"`word_list_filepath` must be a valid .txt file")
    
    word_list_text = get_text_file_content(word_list_filepath)
    word_list = word_list_text.split('\n')

    if not word_list_w_embeddings_filepath.exists():
        logger.info("No cached word list embeddings found. Generating new embeddings. Please be patient...")
        embeddings_dict = generate_embeddings_dict_from_list(word_list, ai_service_caller)
        write_json_to_file(word_list_w_embeddings_filepath, embeddings_dict)
        return embeddings_dict

    try:
        cached_embeddings_dict = get_json_file_content(word_list_w_embeddings_filepath)

        # check count
        if len(word_list) != len(cached_embeddings_dict):
            logger.info("Word list change detected (count mismatch). Regenerating embeddings. Please be patient...")
            embeddings_dict = generate_embeddings_dict_from_list(word_list, ai_service_caller)
            write_json_to_file(word_list_w_embeddings_filepath, embeddings_dict)
            return embeddings_dict

        # check hashes
        current_hashes = [compute_hash(i) for i in word_list]
        cached_hashes = [item.get("text_hash") for item in cached_embeddings_dict]

        if current_hashes != cached_hashes:
            logger.info("Word list change detected (content mismatches). Regenerating embeddings. Please be patient...")
            embeddings_dict = generate_embeddings_dict_from_list(word_list, ai_service_caller)
            write_json_to_file(word_list_w_embeddings_filepath, embeddings_dict)
            return embeddings_dict

        return cached_embeddings_dict

    except Exception as e:
        logger.warning("Failed to validate or parse cached embeddings. Rebuilding. Please be patient...")
        embeddings_dict = generate_embeddings_dict_from_list(word_list, ai_service_caller)
        write_json_to_file(word_list_w_embeddings_filepath, embeddings_dict)
        return embeddings_dict


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two vectors using pure Python.

    Args:
        vec1: First vector (list of floats).
        vec2: Second vector (list of floats).

    Returns:
        Cosine similarity as a float between -1.0 and 1.0.
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of same length")

    dot_prod = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0  # Define similarity with zero magnitude vector as 0

    return dot_prod / (norm1 * norm2)


def get_relevant_word_list_terms(
    word_list_embeddings: List[Dict[str, any]],
    text_passage_embedding: List[float],
    similarity_threshold: float = 0.70
) -> List[str]:
    """
    Return terms from word list where embedding cosine similarity to text_passage_embedding
    is greater than or equal to similarity_threshold.
    """
    matching_terms = []
    for item in word_list_embeddings:
        similarity = cosine_similarity(item["embedding"], text_passage_embedding)
        if similarity >= similarity_threshold:
            matching_terms.append(item["term"])
    return matching_terms


def generate_style_guide_text(
    text_passage: str,
    word_list_filepath: Union[Path, str],
    word_list_w_embeddings_filepath: Union[Path, str],
    style_rules_filepath: Union[Path, str],
    ai_service_caller: AIServiceCaller,
    ):
    """
    Use embeddings of existing style rules, word list, and text passage
    to generate style guide text. 

    Args:
        text_passage: text in prompt to be edited by AI service
        word_list_filepath: filepath to newline delimited word list text file
        word_list_w_embeddings_filepath: file path to JSON file with word list and embeddings
        style_rules_filepath: file path to text file containing other style rules
        ai_service_caller: instance of AIServiceCaller, for creating embeddings

    Return:
        string representing style guide portion of prompt
    """
    word_list_w_embeddings = check_and_update_wordlist_embeddings(word_list_filepath, word_list_w_embeddings_filepath, ai_service_caller)
    text_passage_embedding = ai_service_caller.generate_embedding(input_text=text_passage)
    relevant_word_list_terms = get_relevant_word_list_terms(word_list_w_embeddings, text_passage_embedding)
    style_rules_str = get_text_file_content(style_rules_filepath)
    word_list_str = '=== Word List' + '\n' + '\n'.join(relevant_word_list_terms) if relevant_word_list_terms else ''
    style_guide_text_for_prompt = style_rules_str + word_list_str
    return style_guide_text_for_prompt
