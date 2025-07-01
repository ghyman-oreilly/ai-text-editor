from typing import Dict

from helpers import count_token_length
from models import TextFileFormat


class MaxTokensExceeded(Exception):
	"""Raise when prompt exceeds max token limit"""
	pass

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