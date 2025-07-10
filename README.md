
# AI Editor Script

A script for using AI to edit documents in alignment with an editorial stylesheet and wordlist.

## Requirements

- Python (3.9+ recommended)
- OpenAI API key

## Setup

1. Clone the repository or download the source files:

	```bash
	git@github.com:ghyman-oreilly/ai-text-editor.git
	
	cd ai-text-editor
	```

2. Install required dependencies:

	```bash
	pip install -r requirements.txt
	```

3. Create an `.env` file in the project directory to store your OpenAI API key:

	```bash
	echo "OPENAI_API_KEY=your-key-here" >> .env
	```

## Usage

To edit one or more documents, run the following command:

```bash
python main.py <input_path>
```

Where `<input_path>` is one of the following:

- A path to a JSON file with a 'files' list of such documents (recommended)
- A path to a directory containing documents to be edited
- Space-delimited paths to such documents


Options:
- `--load-data-from-json`, `-l`: Pass this flag with the path to an optional JSON file of data backed up from a previous session. Useful for continuing your progress after a session is interrupted, without having to send all data back to the AI service.
- `--disable-qa-pass`, `-q`: Pass this flag to skip the QA pass of AI service calls, during which the LLM model is prompted to check the edited text against the original, looking for and correcting introduced formatting errors.
- `--model`, `-m`: Pass this flag with your choice of OpenAI model to be used for editing, QA, and global review of the documents. Options are `gpt-4o`, `gpt-4.1`, and `o3`. `gpt-4o` is the default.

NOTE: Because the script rewrites files in place, it's recommended that it be run only on clean Git repos, so the changes can easily be reviewed and reverted, as needed.

## Additional Configuration

This project includes in the `style_guides` folder a local style guide (for passage-level editing), a wordlist (for passage-level editing), and a global style guide (for document-level review), each of which can be edited/amended to influence the model output:

- `style_guide_local.json`: local style guide containing style rules for passage-level editing. Each rule includes fields used to deterministically inject relevant rules into the prompt. A secondary RAG approach is also used to inject additional rules. This file can be edited/amended, but changes in the structure may result in errors.

- `wordlist.txt`: wordlist containing term spellings and style for passage-level editing. RAG is used to inject relevant terms into the editing prompt. This file can be edited/amended: terms should be separated by a newline.

- `style_guide_global.json`: global style guide containing style rules for document-level review. File structure follows that of `style_guide_local.json`, with field values being used to deterministically inject rules into the prompt. This file can be edited/amended, but changes in the structure may result in errors.

After changes are made to `style_guide_local.json` or `wordlist.txt`, an additional process of the script is triggered, whereby files containing the rules and their embeddings (binary `.npz` files in the `style_guides` folder) are regenerated.

## Limitations

* Only Asciidoc file format (`.asciidoc` or `.adoc`) currently supported. 
