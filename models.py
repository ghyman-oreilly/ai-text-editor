import json
from pathlib import Path
from pydantic import BaseModel, Field
import re
from typing import List, Literal, Optional, Union


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


class StyleInsertionCondition(BaseModel):
    format: Literal['asciidoc']
    patterns: Union[List[str]] = []


class StyleRule(BaseModel):
    content: str
    scope: Literal['local', 'global']
    insertion_conditions: Union[List[StyleInsertionCondition], List] = []
    always_insert: bool = False


class StyleCategory(BaseModel):
    name: str = Field(alias="category")
    rules: List[StyleRule]
    insertion_conditions: Union[List[StyleInsertionCondition], List] = []
    always_insert: bool = False


class StyleGuide(BaseModel):
    categories: List[StyleCategory]
    scope: Literal['local', 'global']


class StyleGuide(BaseModel):
    categories: List[StyleCategory]
    scope: Literal['local', 'global']

    def get_matching_rule_contents(
        self,
        input_text: str,
        file_format: Literal['asciidoc'],
    ) -> List[str]:
        """
        Return a list of StyleRule.content values where:
        1. The rule's category has an insertion_condition (matching format) and a pattern that matches input_text
        2. OR the rule itself has such insertion_conditions
        3. OR category.always_insert is True
        4. OR rule.always_insert is True
        """
        matched_rules: List[str] = []

        for category in self.categories:
            category_matches = (
                category.always_insert or
                matches_insertion_conditions(input_text, category.insertion_conditions, file_format)
            )

            for rule in category.rules:
                rule_matches = (
                    rule.always_insert or
                    matches_insertion_conditions(input_text, rule.insertion_conditions, file_format)
                )

                if category_matches or rule_matches:
                    matched_rules.append(rule.content)

        return matched_rules


class Embedding(BaseModel):
    content: str
    embedding: List[float]
    hash_val: str
    model: str


def load_style_guide(path: Union[str, Path], scope: Literal["local", "global"] = "local") -> StyleGuide:
    with open(str(path), 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Wrap the list under the expected root schema
    wrapped_data = {
        "scope": scope,
        "categories": data
    }

    return StyleGuide.model_validate(wrapped_data)

def matches_insertion_conditions(
    input_text: str,
    conditions: List[StyleInsertionCondition],
    file_format: Literal["asciidoc"]
) -> bool:
    for cond in conditions:
        if cond.format != file_format:
            continue
        for pattern in cond.patterns:
            if re.search(pattern, input_text, flags=re.MULTILINE):
                return True
    return False

