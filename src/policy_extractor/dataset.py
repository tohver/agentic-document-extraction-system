"""Read documents and gold labels from disk. A document's id is its file name without suffix."""

from pathlib import Path

from policy_extractor.evaluate import GoldLabel


def load_documents(path: Path) -> dict[str, str]:
    """Load one `.txt` file, or every `.txt` file in a directory, sorted by id."""
    files = [path] if path.is_file() else sorted(path.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no .txt documents found at {path}")
    return {file.stem: file.read_text(encoding="utf-8") for file in files}


def load_gold(directory: Path) -> dict[str, GoldLabel]:
    labels = [
        GoldLabel.model_validate_json(file.read_text(encoding="utf-8"))
        for file in sorted(directory.glob("*.json"))
    ]
    return {label.doc_id: label for label in labels}
