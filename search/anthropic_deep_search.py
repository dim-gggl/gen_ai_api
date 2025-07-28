import anthropic
import argparse
import random
import string
import json

from ..config import ANTHROPIC_API_KEY


client = anthropic.Anthropic(
    api_key=ANTHROPIC_API_KEY
)


def add_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=__file__,
        description="Deep search with Anthropic",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        usage="python3 anthropic_deep_search.py [-p | --prompt=<text>] [-o | --file-name=<path>] [-t | --thinking-budget=<value>] [-m | --max-tokens=<value>] [-M | --model=<value>] <command> [<args>]",
        epilog="Example: python3 anthropic_deep_search.py -p 'What is the capital of France?' -o 'response.txt' -t 10000 -m 16000 -M 'claude-sonnet-4-20250514'",
        add_help=True,
        allow_abbrev=True,
        fromfile_prefix_chars="@",
        argument_default=argparse.SUPPRESS,
        conflict_handler="resolve",
        exit_on_error=False,
        prefix_chars="-"
    )
    parser.add_argument("-p", "-prompt", type=str,  help="The prompt to send to the model")
    parser.add_argument("-o", "-file-name", type=str, help="The name of the file to save the response to")
    parser.add_argument("-t", "-thinking-budget", type=int, default=10000, help="The budget of tokens for the thinking")
    parser.add_argument("-m", "-max-tokens", type=int, default=16000, help="The maximum number of tokens for the response")
    parser.add_argument("-M", "--model", type=str, default="claude-sonnet-4-20250514", help="The model to use")

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    if not args.prompt:
        prompt = input("Enter the prompt: ").strip()
    else:
        prompt = args.prompt
    thinking_budget = args.thinking_budget
    max_tokens = args.max_tokens
    model = args.model

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        thinking={
            "type": "enabled",
            "budget_tokens": thinking_budget
        },
        messages=[{
            "role": "user",
            "content": { "input_text": prompt }
        }]
    )

    reasoning_blocks = []
    text_blocks = []
    for block in response.content:
        if block.type == "thinking":
            reasoning_blocks.append(block.thinking)
            print(f"\nThinking : {block.thinking}")
        elif block.type == "text":
            text_blocks.append(block.text)
        
        print(f"\n{block.text}")

        print(
            f"{f'[INFO]':^90}\n{__file__} : {__name__} - l51 : \n{block=}")
    if not args.file_name:
        file_name = f"{args.prompt[:10]}_{random.choices(string.ascii_letters, k=10)}"
    else:
        file_name = args.file_name

    with open(f"text/output/RAW_{file_name}.txt", "w") as f:
        f.write(response.content)
    
    response_json = {
        "prompt": prompt,
        "model": model,
        "thinking_budget": thinking_budget,
        "max_tokens": max_tokens,
        "reasoning_blocks": [
            b for b in reasoning_blocks
        ],
        "text_blocks": [
            b for b in text_blocks
        ]
    }

    with open(f"text/output/{file_name}.json", "w") as f:
        json.dump(response_json, f, ensure_ascii=True,indent=4)

    

if __name__ == "__main__":
    args = add_args()
    main(args)