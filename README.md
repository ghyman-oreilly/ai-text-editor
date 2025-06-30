
# AI Editor Script

A script for using AI to edit documents in alignment with an editorial stylesheet.

## Limitations

* Only Asciidoc file format (.asciidoc or .adoc) currently supported.

## TODO

* Tokenize style guide and prompt instructions, and set max_tokens_per_block in extract_ascii_blocks based on remaining threshold (want to keep it small for focus), keeping selected model in mind

* Set model in CLI and emit out to read_files and ai_service

* Consider: for making text replacements, do we want to use regex search in combination with index tracking (the latter b/c there could be identical text blocks)?