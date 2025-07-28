#!/usr/bin/env python3
import os
import argparse
from openai import OpenAI

from configs import OPENAI_API_KEY, OPENAI_MODELS
from utils import encode_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate text via OpenAI Chat Completions API"
    )
    parser.add_argument("--api_key",
                        default=OPENAI_API_KEY,
                        help="OpenAI API key (env: OPENAI_API_KEY)")
    parser.add_argument("--api_base",
                        default=os.getenv("OPENAI_BASE_URL", 
                                          "https://api.openai.com/v1"),
                        help="Custom API base URL if using a proxy or enterprise endpoint")
    parser.add_argument("--organization",
                        default=os.getenv("OPENAI_ORGANIZATION"),
                        help="Organization ID for OpenAI enterprise users")
    parser.add_argument("--timeout",
                        type=float,
                        default=600.0,
                        help="Timeout for requests, in seconds (default 10 minutes)")
    parser.add_argument("--max_retries",
                        type=int,
                        default=2,
                        help="Number of automatic retries for transient errors")
    parser.add_argument("--model",
                        default=OPENAI_MODELS[0],
                        help="Chat model to use")
    parser.add_argument("--prompt",
                        default=input("Enter your prompt: ").strip(),
                        help="Prompt to send as the user message")
    parser.add_argument("--stream",
                        action="store_true",
                        help="Stream responses as they arrive")
    parser.add_argument("--input-image",
                        help="Path to the image to pass as input")
    parser.add_argument("--input-file",
                        help="Path to the file to pass as input")
    args = parser.parse_args()

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
        organization=args.organization,
        timeout=args.timeout,
        max_retries=args.max_retries
    )  # :contentReference[oaicite:5]{index=5}

    messages = [{"role": "user", "content": [args.prompt]}]
    if args.input_image:
        encoded_image = encode_file(args.input_image)
    if args.input_file:
        encoded_file = encode_file(args.input_file)

    response = client.responses.create(
        model=args.model,
        input=[
            {"role": "user",
                "content": [
                    { "type": "input_text", "text": args.prompt },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                ],
            }
        ],
        background=False,
        include=[],

    )

    print(response)


    # resp = client.chat.completions.create(
    #     model=args.model,
    #     messages=messages,
    #     stream=args.stream
    # )  # :contentReference[oaicite:6]{index=6}

    # if args.stream:
    #     for event in resp:
    #         print(event.choices[0].message.content, end="")
    # else:
    #     print(resp.choices[0].message.content)

if __name__ == "__main__":
    main()
