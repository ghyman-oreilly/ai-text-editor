import hashlib
import logging
import math
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from embeddings import filter_by_vector_similarity
from helpers import count_token_length, get_text_file_content
from models import Embedding


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

GLOBAL_REVIEW_PROMPT_BASE_TEXT = """
You are an expert copyeditor. Review the passage below for problems of consistency or style, based on the accompanying style guide. Do not summarize or restate style rules unless the passage actively violates them. Only report issues that appear directly in the quoted text.

[BEGIN STYLE GUIDE]

{style_guide}

[END STYLE GUIDE]

[BEGIN PASSAGE TO REVIEW]

{passage_to_be_reviewed}

[END PASSAGE TO REVIEW]

Present your output as a list of issues in the following Markdown format:

Use a numbered list.
Each item should have:
A bolded brief title.
A short explanation of the issue.
A section labeled Original Text: containing the exact sentence or phrase from the passage that exhibits the problem.
An optional section labeled Suggested Text: with your suggested correction.
Do not include entries for rules that aren’t violated in the passage.

If no issues are found in the passage, respond only with: NO_ISSUE
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
        Fully formatted prompt string, ready for LLM input, or None.
    """
    try:
        prompt_text = prompt_template.format(**template_kwargs)
    except KeyError as e:
        raise ValueError(f"Missing a required template field: {e}")

    tokens = count_token_length(prompt_text, model)

    if tokens > max_tokens_per_prompt:
        logger.warning(f"Prompt contains {tokens} tokens, which exceeds model limit of {max_tokens_per_prompt}")
        return None
    return prompt_text


def generate_style_guide_text(
    text_passage: str,
    word_list_embeddings: List[Embedding],
    local_style_rules_embeddings: List[Embedding],
    embed_function: Callable,
    other_style_rules_to_inject: Union[List[str], List] = []
    ):
    """
    Use embeddings of existing style rules, word list, and text passage
    to generate style guide text. 

    Return:
        string representing style guide portion of prompt
    """
    text_passage_embedding = embed_function(text_passage)
    relevant_word_list_terms = [i for i in filter_by_vector_similarity(word_list_embeddings, text_passage_embedding)]
    relevant_style_rules = [i for i in filter_by_vector_similarity(local_style_rules_embeddings, text_passage_embedding)]
    if other_style_rules_to_inject:
         relevant_style_rules = list(set(relevant_style_rules) | set(other_style_rules_to_inject))
    style_rules_str = '=== Style Rules' + '\n' + '\n'.join(relevant_style_rules) if relevant_style_rules else ''
    word_list_str = '=== Word List' + '\n' + '\n'.join(relevant_word_list_terms) if relevant_word_list_terms else ''
    style_guide_text_for_prompt = style_rules_str + '\n' + word_list_str
    return style_guide_text_for_prompt
