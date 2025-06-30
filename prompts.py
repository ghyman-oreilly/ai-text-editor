from helpers import count_token_length

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

Preserve the formatting and markup style (e.g., AsciiDoc, HTML, Markdown, etc.) used in the input.

Do not add language tags or triple backticks (```).
"""

def generate_copyedit_prompt_text(
		text_to_be_edited: str, 
		preceding_passage_text: str, 
		style_guide_text: str,
		model: str = "gpt-4o",
		max_tokens_per_prompt: int = 20000
	):
	prompt_text = COPYEDIT_PROMPT_BASE_TEXT.format(style_guide=style_guide_text, preceding_passage=preceding_passage_text, passage_to_be_edited=text_to_be_edited)
	tokens = count_token_length(prompt_text, model)
	if tokens > max_tokens_per_prompt:
		raise MaxTokensExceeded
	return prompt_text
