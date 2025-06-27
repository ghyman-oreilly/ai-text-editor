from pathlib import Path
from pydantic import basemodel
from typing import List, Optional


class TextBlock(basemodel):
	index: int # order of appearance in file
	file_id: str
	block_id: str
	original_content: str
	ai_edited_content: str
	is_processed: bool = False


class TextFile(basemodel):
	id: str
	filepath: Path
	text_blocks: Optional[List[TextBlock]] = []

	@property
	def is_fully_processed(self):
		return all([b.is_processed for b in self.text_blocks])

	@property
	def original_content(self):
		return '\n'.join([b.original_content for b in self.text_blocks])

	@property
	def ai_edited_content(self):
		return '\n'.join([b.ai_edited_content or b.original_content for b in self.text_blocks])


class AsciiBlock(TextBlock):
	pass


class AsciiFile(TextFile):
	pass

