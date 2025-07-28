import anthropic
import argparse
import random
import string
import json

from configs import ANTHROPIC_API_KEY
from cli_core import command, set_build_parser


@command('anthropic-deep-search', help='Deep search with Anthropic')
def anthropic_deep_search_main(args):
    main(args)


def main(args):
    prompt = args.prompt
    if not prompt:
        prompt = input("Enter the prompt: ").strip()
    else:
        prompt = args.prompt

    client = anthropic.Anthropic(
        api_key=ANTHROPIC_API_KEY
    )

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
            "content": [
                { "type": "text", "text": prompt }
            ]
        }]
    )

    reasoning_blocks = []
    text_blocks = []
    for block in response.content:
        if block.type == "thinking":
            reasoning_blocks.append(block.thinking)
            print(f"\nThinking : {block.thinking}")
        elif block.type == "text":
            text_blocks.append(block)
        
            print(f"\n{block}")

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

@set_build_parser('anthropic-deep-search')
def build(p):
    p.add_argument("--prompt", type=str,  help="The prompt to send to the model")
    p.add_argument("--file-name", type=str, help="The name of the file to save the response to")
    p.add_argument("--thinking-budget", type=int, default=10000, help="The budget of tokens for the thinking")
    p.add_argument("--max-tokens", type=int, default=16000, help="The maximum number of tokens for the response")
    p.add_argument("--model", type=str, default="claude-sonnet-4-20250514", help="The model to use")
    args = p.parse_args()
    return args

if __name__ == "__main__":
    anthropic_deep_search_main(build(argparse.ArgumentParser()))