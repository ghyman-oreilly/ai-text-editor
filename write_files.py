import logging
from typing import List

from helpers import write_text_to_file
from models import AsciiFile


logger = logging.getLogger(__name__)


def write_files(text_files: List[AsciiFile]):
	"""
	For list of AsciiFiles, write edited text
	to file.
	"""
	for text_file in text_files:
		filepath = text_file.filepath
		edited_text = text_file.get_edited_content()
		write_text_to_file(filepath, edited_text)
		logger.info(f"Edited text written to file: {filepath}")
