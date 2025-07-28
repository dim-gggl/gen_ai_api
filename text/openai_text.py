#!/usr/bin/env python3
import os
import argparse
from openai import OpenAI

from configs import OPENAI_API_KEY, OPENAI_MODELS
from utils import encode_file
from cli_core import command, set_build_parser


@command('openai-text', help='Generate text via OpenAI Chat Completions API')
def openai_text_main(args):
    main(args)

def main(args):
    prompt = args.prompt
    if not prompt:
        prompt = input("Enter your prompt: ").strip()
    
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
        organization=args.organization,
        timeout=args.timeout,
        max_retries=args.max_retries
    )  

    messages = [{"role": "user", "content": prompt}]
    
    # Gestion des images si fournies
    if args.input_image:
        encoded_image = encode_file(args.input_image)
        messages[0]["content"] = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                }
            }
        ]

    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        stream=args.stream
    )

    if args.stream:
        for chunk in response:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="")
    else:
        print(response.choices[0].message.content)

@set_build_parser('openai-text')
def build(p):
    p.add_argument("--api_key",
                    default=OPENAI_API_KEY,
                    help="OpenAI API key (env: OPENAI_API_KEY)")
    p.add_argument("--api-base",
                    default=os.getenv("OPENAI_BASE_URL", 
                                      "https://api.openai.com/v1"),
                    help="Custom API base URL if using a proxy or enterprise endpoint")
    p.add_argument("--organization",
                    default=os.getenv("OPENAI_ORGANIZATION"),
                    help="Organization ID for OpenAI enterprise users")
    p.add_argument("--timeout",
                    type=float,
                    default=600.0,
                    help="Timeout for requests, in seconds (default 10 minutes)")
    p.add_argument("--max_retries",
                    type=int,
                    default=2,
                    help="Number of automatic retries for transient errors")
    p.add_argument("--model",
                    default=OPENAI_MODELS[0],
                    help="Chat model to use")
    p.add_argument("--prompt",
                    help="Prompt to send as the user message")
    p.add_argument("--stream",
                    action="store_true",
                    help="Stream responses as they arrive")
    p.add_argument("--input-image",
                    help="Path to the image to pass as input")
    p.add_argument("--input-file",
                    help="Path to the file to pass as input")

    args = p.parse_args()
    return args

if __name__ == "__main__":
    openai_text_main(build(argparse.ArgumentParser()))
