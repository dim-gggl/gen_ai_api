import base64
import argparse

# Function to encode the image
def encode_file(file_path):
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")

def parse_args(list_of_args, p):
    """Parse arguments from a list of argument specifications.
    
    Args:
        list_of_args: List of dicts with argument specifications
        p: ArgumentParser instance
    
    Returns:
        Parsed arguments namespace
    """
    for arg_spec in list_of_args:
        # Handle the special case where flags are specified as a tuple
        if "flags" in arg_spec:
            flags = arg_spec.pop("flags")
            p.add_argument(*flags, **arg_spec)
        else:
            p.add_argument(**arg_spec)
    return p.parse_args()

# Example usage for standalone scripts
if __name__ == "__main__":
    list_of_args = [
        {"flags": ("--prompt", "-P"), "help": "Prompt to generate the voice"},
        {"flags": ("--voice", "-V"), "help": "Voice to generate the voice"},
        {"flags": ("--model", "-M"), "help": "Model to generate the voice"},
        {"flags": ("--output", "-O"), "help": "Output file path"},
    ]
    p = argparse.ArgumentParser()
    args = parse_args(list_of_args, p)
    print(args)