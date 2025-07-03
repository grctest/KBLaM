import json
from pathlib import Path
from typing import Any

def write_to_json(
    data: Any, filepath: str, indent: int = 4, encoding: str = "utf-8"
) -> bool:
    """
    Write a dictionary to a JSON file with error handling and formatting options.

    Args:
        data: Dictionary to write to JSON file
        filepath: Path where the JSON file should be saved
        indent: Number of spaces for indentation (default: 4)
        encoding: File encoding (default: 'utf-8')

    Raises:
        TypeError: If data is not a dictionary
    """

    try:
        # Convert string path to Path object
        file_path = Path(filepath)

        # Write the JSON file
        with open(file_path, "w", encoding=encoding) as f:
            json.dump(
                data,
                f,
                indent=indent,
                sort_keys=True,  # For consistent output
                default=str,  # Handle non-serializable objects by converting to string
            )

    except Exception as e:
        print(f"Error writing JSON file: {str(e)}")
