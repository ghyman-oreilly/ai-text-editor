from pathlib import Path
from pydantic import BaseModel
from typing import List, Literal, Optional


TextFileFormat = Literal["asciidoc"]


class TextBlock(BaseModel):
	index: int # order of appearance in file
	file_id: str
	block_id: str
	original_content: str
	ai_edited_content: str = ''
	ai_qaed_content: str = ''
	is_edited: bool = False
	is_qaed: bool = False


class TextFile(BaseModel):
	index: int # order of appearance in workflow
	file_format: TextFileFormat
	id: str
	filepath: Path
	text_blocks: Optional[List[TextBlock]] = []

	@property
	def is_fully_processed(self):
		"""
		Check whether all text blocks in file are processed
		"""
		return all([b.is_edited for b in self.text_blocks])

	@property
	def original_content(self):
		"""
		Get original content for the entire file
		"""
		return '\n'.join([b.original_content for b in self.text_blocks])

	def get_qaed_edited_content(self):
		"""
		Get AI-edited content for the entire file, falling back to
		original content if text block not yet edited.

		Making this a class method instead of a property, to help
		prevent confusion in business logic prior to completing AI 
		editing/review (i.e., we won't call the method until that
		process is completed).

		We fall back to original content if no ai_qaed or ai_edited content,
		because some text blocks may not receive edits.
		"""
		return '\n\n'.join([b.ai_qaed_content or b.ai_edited_content or b.original_content for b in self.text_blocks])


class AsciiBlock(TextBlock):
	pass


class AsciiFile(TextFile):
	file_format: TextFileFormat = "asciidoc"
