import os
import json


def count_code_lines_in_notebook(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    code_line_count = 0
    # Iterate through the cells in the notebook
    for cell in notebook.get("cells", []):
        # Only count lines in code cells
        if cell.get("cell_type") == "code":
            # Count the number of lines in the cell's 'source' content
            code_line_count += sum(1 for line in cell.get("source", []) if line.strip())

    return code_line_count


def count_code_lines_in_python_file(py_file_path):
    code_line_count = 0
    with open(py_file_path, "r", encoding="utf-8") as f:
        for line in f:
            # Count non-empty lines
            if line.strip():
                code_line_count += 1
    return code_line_count


def count_code_lines_in_repository(repo_path):
    total_code_lines = 0
    # Walk through all files in the repository
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            # Only process files with .ipynb extension
            if file.endswith(".ipynb"):
                notebook_path = os.path.join(root, file)
                total_code_lines += count_code_lines_in_notebook(notebook_path)
            elif file.endswith('.py'):
                # Count lines in Python files
                py_file_path = os.path.join(root, file)
                total_code_lines += count_code_lines_in_python_file(py_file_path)
    return total_code_lines


# Path to the repository
repo_path = (
    "."  # Change this to the path of your repository if not running from the root
)

# Count and print the total lines of code in the repository
total_code_lines = count_code_lines_in_repository(repo_path)
print(f"Total lines of code in Jupyter notebooks: {total_code_lines}")
