#!/usr/bin/env python3
"""
Query Grok 4 with Live Search enabled and get a long, structured answer.
"""
import argparse, asyncio, os, textwrap
import openai

from cli_core import command, set_build_parser


@command('xai-search', help='Search with xAI')
def xai_search_main(args):
    main(args)

def main(args):
    prompt = args.prompt
    if not prompt:
        prompt = input("Enter your prompt :").strip() or "Fais-moi un état des lieux complet des tendances IA en Europe, structuré en sections."

    client = openai.OpenAI(api_key=args.api_key)

    response = client.chat.completions.create(
        stream=True,
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        search_parameters={"mode": args.search_mode},  # Live Search on :contentReference[oaicite:1]{index=1}
        tools=[{"type": "live_search"}],
        tool_choice="auto",
        response_format={"type": "json_object"},
    )
    content = ""
    for chunk in response:
        content += chunk.choices[0].delta.content  # type: ignore
        print(chunk.choices[0].delta.content, end="", flush=True)  # type: ignore
    print()
    with open("response.json", "w") as f:
        f.write(content)

@set_build_parser('xai-search')
def build(p):
    p.add_argument("--prompt")
    p.add_argument("--model", default="grok-4-online")
    p.add_argument("--max_tokens", type=int, default=16000)

    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--search_mode", choices=["off", "auto", "on"], default="on")
    p.add_argument("--api_key", default=os.getenv("XAI_API_KEY"))
    p.add_argument("--api_host", default="api.x.ai")
    args = p.parse_args()
    return args

if __name__ == "__main__":
    xai_search_main(build(argparse.ArgumentParser()))

