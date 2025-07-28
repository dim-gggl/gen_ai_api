#!/usr/bin/env python3
import argparse
from google import genai
from google.genai import types

from configs import GOOGLE_MODELS, GOOGLE_API_KEY
from cli_core import command, set_build_parser


@command('gemini-text', help='Generate text via Google Gemini API')
def gemini_text_main(args):
    # Build HTTP options if a specific API version is requested
    http_opts = types.HttpOptions(api_version=args.api_version) if args.api_version else None

    client_kwargs = {
        "api_key": args.api_key,
        "vertexai": args.vertexai,
        "project": args.project,
        "location": args.location,
    }
    if http_opts:
        client_kwargs["http_options"] = http_opts

    client = genai.Client(**{k: v for k, v in client_kwargs.items() if v is not None})
    

    # Configure generation settings
    gen_config = types.GenerateContentConfig(
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
    ) if (args.max_output_tokens or args.temperature) else None

    response = client.models.generate_content(
        model=args.model,
        contents=args.prompt,
        config=gen_config
    )  

    print(response.text)

@set_build_parser('gemini-text')
def build(p):
    p.add_argument("--api-key",
                    default=GOOGLE_API_KEY,
                    help="Gemini API key (env: GEMINI_API_KEY or GOOGLE_API_KEY)")
    p.add_argument("-M",
                    "--model",
                    choices=GOOGLE_MODELS,
                    default="gemini-2.5-flash",
                    help="Gemini model to use")
    p.add_argument("--prompt",
                    default=input("Enter your prompt: ").strip(),
                    help="Text prompt for content generation")
    p.add_argument("--max-output-tokens",
                    type=int,
                    default=1024,
                    help="Maximum number of tokens to generate")
    p.add_argument("--temperature",
                    type=float,
                    default=0.0,
                    help="Sampling temperature (0.0 - 1.0)")
    args = p.parse_args()
    return args

if __name__ == "__main__":
    gemini_text_main(build(argparse.ArgumentParser()))
