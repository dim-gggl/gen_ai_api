#!/usr/bin/env python3
import os
import argparse
import asyncio
import xai_sdk

from configs import XAI_MODELS

async def main():
    parser = argparse.ArgumentParser(
        description="Generate text via xAI Grok API"
    )
    parser.add_argument("--api_key",
                        default=os.getenv("XAI_API_KEY"),
                        help="xAI API key (env: XAI_API_KEY)")
    parser.add_argument("--api_host",
                        default="api.x.ai",
                        help="Hostname of the xAI API server")  # :contentReference[oaicite:0]{index=0}
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="Seed for deterministic sampling")
    parser.add_argument("--mode",
                        choices=["sample", "chat"],
                        default="sample",
                        help="Mode: 'sample' for raw token sampling or 'chat' for stateless chat")
    parser.add_argument("--prompt",
                        default="Hello, Grok!",
                        help="Prompt text to send to the API")
    parser.add_argument("--max_len",
                        type=int,
                        default=50,
                        help="Maximum number of tokens to generate (for sampler)")
    parser.add_argument("--model",
                        choices=XAI_MODELS,
                        default="grok-3-mini-fast",
                        help="Model to use")
    parser.add_argument("--temperature",
                        type=float,
                        default=0.5,
                        help="Temperature for sampling, 0.0 is deterministic, 1.0 is random")
    parser.add_argument("--top_p",
                        type=float,
                        default=1.0,
                        help="Top-p for sampling")
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of completions to generate")
    args = parser.parse_args()

    client = xai_sdk.Client(
        api_key=args.api_key,
        initial_rng_seed=args.initial_rng_seed,
        api_host=args.api_host
    )  # :contentReference[oaicite:1]{index=1}

    if args.mode == "sample":
        # Asynchronously stream raw tokens
        async for token in client.sampler.sample(
            args.prompt,
            max_len=args.max_len
        ):  # :contentReference[oaicite:2]{index=2}
            print(token.token_str, end="")
        print()
    else:
        # Stateless chat completion
        conv = client.chat.create_conversation()
        response = await conv.add_response_no_stream(args.prompt)
        print(response.message)

if __name__ == "__main__":
    asyncio.run(main())
