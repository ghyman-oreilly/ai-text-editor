from pydantic import BaseModel
from typing import List, Literal, Union


class StyleInsertionCondition(BaseModel):
	format: Literal['asciidoc']
	patterns: Union[List[str]] = []


class StyleRule(BaseModel):
	content: str
	scope: Literal['local', 'global']
	insertion_conditions: Union[List[StyleInsertionCondition], List] = []
	always_insert: bool = False


class StyleCategory(BaseModel):
	name: str
	rules: List[StyleRule]
	insertion_conditions: Union[List[StyleInsertionCondition], List] = []
	always_insert: bool = False


class StyleGuide(BaseModel):
	categories: List[StyleCategory]
	scope: Literal['local', 'global']
