#!/usr/bin/env python3
"""
Ground Gemini with Google Search and return a cited, structured answer.
"""
import argparse, os, textwrap
from google import genai
from google.genai import types

from cli_core import command, set_build_parser


@command('gemini-search', help='Search with Gemini')
def gemini_search_main(args):
    main(args)

def main(args):
    prompt = args.prompt
    if not prompt:
        prompt = input("Enter your prompt :").strip() or "Fais-moi un état des lieux complet des tendances IA en Europe, structuré en sections."

    client = genai.Client(api_key=args.api_key or os.getenv("GEMINI_API_KEY"))  # :contentReference[oaicite:3]{index=3}

    # Nouveau tool 'google_search' (v2.x) — sinon utiliser google_search_retrieval (legacy) :contentReference[oaicite:4]{index=4}
    search_tool = types.Tool(google_search=types.GoogleSearchConfig(
        search_type=types.GoogleSearchConfig.SearchType.WEB,
        max_results=10,
    ))
    cfg = types.GenerateContentConfig(
        tools=[search_tool],
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )

    resp = client.models.generate_content(
        model=args.model,
        contents=prompt,
        config=cfg,
    )
    return resp.text

@set_build_parser('gemini-search')
def build(p):
    ap = argparse.ArgumentParser(description="Gemini with Google Search grounding")
    p.add_argument("--prompt")
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max_output_tokens", type=int, default=1024)
    ap.add_argument("--api_key", default=os.getenv("GEMINI_API_KEY"))
    args = ap.parse_args()
    print(run_query(**vars(args)))

if __name__ == "__main__":
    main()
