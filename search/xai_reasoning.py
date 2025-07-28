import os
import httpx
import argparse
from openai import OpenAI

from cli_core import command, set_build_parser


@command('xai-reasoning', help='Reasoning with xAI')
def xai_reasoning_main(args):
    main(args)

def main(args):
    prompt = args.prompt
    if not prompt:
        prompt = input("Enter your prompt :").strip()
    
    messages = [
        {
            "role": "system",
            "content": "You are a highly intelligent AI assistant.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    client = OpenAI(
        base_url=args.api_host,
        api_key=args.api_key,
        timeout=httpx.Timeout(args.timeout),  # Override default timeout with longer timeout for reasoning models
    )

    completion = client.chat.completions.create(
        model=args.model,
        messages=messages,
    )

    print("Reasoning Content:")
    print(completion.choices[0].message.reasoning_content)
    reasoning_content = completion.choices[0].message.reasoning_content

    print("Final Response:")
    print(completion.choices[0].message.content)
    content = completion.choices[0].message.content

    print("Number of completion tokens:")
    print(completion.usage.completion_tokens)
    completion_tokens = completion.usage.completion_tokens

    print("Number of reasoning tokens:")
    print(completion.usage.completion_tokens_details.reasoning_tokens)
    reasoning_tokens = completion.usage.completion_tokens_details.reasoning_tokens

    print("Raw")
    print(completion.model_dump_json(indent=4)re)
    invoiced_data = {
        "reasoning_content": reasoning_content,
        "content": content,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "raw": completion
    }
    file_path = args.output or f"search/output/xai_reasoning_{completion.id}_{args.model}.json"
    with open(file_path, "w") as f:
        import json
        json.dump(invoiced_data, f, ensure_ascii=True, indent=4)
    print(f"Invoiced data saved to {file_path}")

@set_build_parser('xai-reasoning')
def build(p):
    p.add_argument("-p", "--prompt")
    p.add_argument("-o", "--output", help="Output file path")
    p.add_argument("-M", "--model", default="grok-3-mini-fast", help="Model to use", choices=["grok-4", "grok-3-mini", "grok-3-mini-fast"])
    p.add_argument("-k", "--api_key", default=os.getenv("XAI_API_KEY"))
    p.add_argument("-t", "--timeout", type=float, default=3600.0, help="Timeout for the http request in seconds")
    p.add_argument("-a", "--api_host", default="https://api.x.ai/v1", help="API host")
    args = p.parse_args()
    return args

if __name__ == "__main__":
    xai_reasoning_main(build(argparse.ArgumentParser()))