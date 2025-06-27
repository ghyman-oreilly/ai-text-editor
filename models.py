from pathlib import Path
from pydantic import basemodel
from typing import List, Literal, Optional


TextFileFormat = Literal["asciidoc"]


class TextBlock(basemodel):
	index: int # order of appearance in file
	file_id: str
	block_id: str
	original_content: str
	ai_edited_content: str = ''
	is_processed: bool = False


class TextFile(basemodel):
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
		return all([b.is_processed for b in self.text_blocks])

	@property
	def original_content(self):
		"""
		Get original content for the entire file
		"""
		return '\n'.join([b.original_content for b in self.text_blocks])

	@property
	def ai_edited_content(self):
		"""
		Get AI-edited content for the entire file, falling back to
		original content if text block not yet edited
		"""
		return '\n'.join([b.ai_edited_content or b.original_content for b in self.text_blocks])


class AsciiBlock(TextBlock):
	pass


class AsciiFile(TextFile):
	file_format: TextFileFormat = "asciidoc"
