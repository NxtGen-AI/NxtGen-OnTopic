"""
This module provides functions to format and print logs and lists in a visually
structured and readable manner.

Key functionalities:
- log_with_horizontal_line(message): Prints a message surrounded by horizontal
lines, making it stand out.
- log_ordered_list_with_horizontal_break(message, collection): Displays an ordered
list with proper indentation for multi-line items, enhancing readability.

These functions are designed to improve the clarity and organization of console
output, making logs and lists easier to read and interpret.
"""
import os

def log_with_horizontal_line(message):
    """Function to print logs in a structured and readable format"""
    # Get the terminal width
    terminal_width = os.get_terminal_size().columns

    # Create the horizontal line
    horizontal_line = '-' * terminal_width

    # Print the formatted message
    print(horizontal_line)
    print()
    print(message)
    print()
    print(horizontal_line)

def log_ordered_list_with_horizontal_break(message: str, collection: list):
    """Function to print ordered lists in a structured and readable format"""
    # Get the terminal width
    terminal_width = os.get_terminal_size().columns

    # Create the horizontal line and indent
    horizontal_line = '-' * terminal_width

    # Print the formatted message
    print(horizontal_line)
    print()
    print(message)
    for i, item in enumerate(collection, 1):
        lines = item.split("\\n")
        print(f"{i}. {lines[0]}")  # Print the number and first line together
        for line in lines[1:]:       # Indent subsequent lines
            print(f"    {line}")
        print()
    print()
    print(horizontal_line)