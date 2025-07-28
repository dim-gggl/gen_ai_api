#!/usr/bin/env python3
import os
import argparse
from google import genai
from google.genai import types

from configs import GOOGLE_MODELS, GOOGLE_API_KEY

def main():
    parser = argparse.ArgumentParser(
        description="Generate text via Google Gemini API"
    )
    parser.add_argument("--api_key",
                        default=GOOGLE_API_KEY,
                        help="Gemini API key (env: GEMINI_API_KEY or GOOGLE_API_KEY)")
    parser.add_argument("--vertexai",
                        action="store_true",
                        help="Use Vertex AI endpoints instead of Gemini Developer API")
    parser.add_argument("--project",
                        default=os.getenv("GOOGLE_CLOUD_PROJECT"),
                        help="Google Cloud Project ID (required for Vertex AI)")
    parser.add_argument("--location",
                        default=os.getenv("GOOGLE_CLOUD_LOCATION"),
                        help="Google Cloud location (e.g. 'us-central1')")
    parser.add_argument("--api_version",
                        default=None,
                        help="API version (v1, v1beta3, etc.)")
    parser.add_argument("--model",
                        default="gemini-2.5-flash",
                        help="Gemini model to use")
    parser.add_argument("--prompt",
                        default=input("Enter your prompt: ").strip(),
                        help="Text prompt for content generation")
    parser.add_argument("--max_output_tokens",
                        type=int,
                        default=None,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature",
                        type=float,
                        default=None,
                        help="Sampling temperature")
    args = parser.parse_args()

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
    # :contentReference[oaicite:3]{index=3}

    # Configure generation settings
    gen_config = types.GenerateContentConfig(
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
    ) if (args.max_output_tokens or args.temperature) else None

    response = client.models.generate_content(
        model=args.model,
        contents=args.prompt,
        config=gen_config
    )  # :contentReference[oaicite:4]{index=4}

    print(response.text)

if __name__ == "__main__":
    main()
