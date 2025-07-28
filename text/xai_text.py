#!/usr/bin/env python3
import os
import argparse
import asyncio
import openai

from configs import XAI_MODELS
from cli_core import command, set_build_parser


@command('xai-text', help='Generate text via xAI Grok API')
def xai_text_main():
    args = build(argparse.ArgumentParser())
    asyncio.run(main(args))
    
async def main(args):
    prompt = args.prompt
    if not prompt:
        prompt = input("Enter your prompt: ").strip()

    client = openai.OpenAI(    
        api_key=args.api_key,
        api_host=args.api_host
    )  

    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=args.temperature,
        top_p=args.top_p,
        n=args.n
    )
    print(response.choices[0].message.content)

@set_build_parser('xai-text')
def build(p):
    p.add_argument("--api-key",
                    default=os.getenv("XAI_API_KEY"),
                    help="xAI API key (env: XAI_API_KEY)")
    p.add_argument("--api-host",
                    default="api.x.ai",
                    help="Hostname of the xAI API server")  
    p.add_argument("--seed",
                    type=int,
                    default=None,
                    help="Seed for deterministic sampling")
    p.add_argument("--mode",
                    choices=["sample", "chat"],
                    default="sample",
                    help="Mode: 'sample' for raw token sampling or 'chat' for stateless chat")
    p.add_argument("--prompt",
                    help="Prompt text to send to the API")
    p.add_argument("--max-len",
                    type=int,
                    default=50,
                    help="Maximum number of tokens to generate (for sampler)")
    p.add_argument("--model",
                    choices=XAI_MODELS,
                    default="grok-3-mini-fast",
                    help="Model to use")
    p.add_argument("--temperature",
                    type=float,
                    default=0.5,
                    help="Temperature for sampling, 0.0 is deterministic, 1.0 is random")
    p.add_argument("--top-p",
                    type=float,
                    default=1.0,
                    help="Top-p for sampling")
    p.add_argument("--n",
                    type=int,
                    default=1,
                    help="Number of completions to generate")
    args = p.parse_args()
    return args

if __name__ == "__main__":
    xai_text_main()