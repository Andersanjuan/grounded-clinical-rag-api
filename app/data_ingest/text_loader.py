from pathlib import Path
from typing import List, Dict

def load_text_files(directory: str) -> List[Dict]:
    """
    Load text files from the specified directory and return their contents
    along with metadata.

    Args:
        directory (str): The path to the directory containing text files.

    Returns:
        List[Dict]: A list of dictionaries, each containing 'content' and 'metadata'.
    """
    base_path = Path(directory)
    docs = []

    for path in base_path.glob("*.txt"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs.append(
            {
                "content": text,
                "metadata": {
                    "source": str(path),
                    "filename": path.name,
                },
            }
        )

    return docs