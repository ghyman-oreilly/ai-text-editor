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

Preserve all formatting and markup consistent with {format_type}. Do not assume, infer, or substitute any other markup style under any circumstances. Leave all formatting exactly as it appears in the input. Your only task is to edit the text for grammar, clarity, and style, in accordance with the style guide.

Do NOT enclose your response in triple backticks (```) or add a language tag.
"""

def generate_copyedit_prompt_text(
		text_to_be_edited: str, 
		preceding_passage_text: str, 
		format_type: TextFileFormat,
		style_guide_text: str,
		model: str = "gpt-4o",
		max_tokens_per_prompt: int = 20000
	):
	"""
	Build prompt text for text-block copyediting AI service call.
	"""
	prompt_text = COPYEDIT_PROMPT_BASE_TEXT.format(style_guide=style_guide_text, preceding_passage=preceding_passage_text, format_type=format_type, passage_to_be_edited=text_to_be_edited)
	tokens = count_token_length(prompt_text, model)
	if tokens > max_tokens_per_prompt:
		raise MaxTokensExceeded
	return prompt_text
