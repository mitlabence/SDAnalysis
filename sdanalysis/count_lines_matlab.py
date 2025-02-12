import os

def count_lines_of_code(file_path):
    """Count the number of lines of code in a given file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return len(lines)

def search_and_count(folder_path):
    """Search for .m files in the given folder and count lines of code."""
    total_lines = 0
    
    # Iterate over the found .m files and count lines of code
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.m'):
                file_path = os.path.join(root, file)
                lines = count_lines_of_code(file_path)
                print(f"{file_path}: {lines} lines")
                total_lines += lines
        
    
    print(f"Total lines of code: {total_lines}")

folder_path = '.'
# Run the search and count function
search_and_count(folder_path)
