#!/usr/bin/env python3
"""
Ground Gemini with Google Search and return a cited, structured answer.
"""
import argparse, os, textwrap
from google import genai
from google.genai import types

DEFAULT_PROMPT = input("Enter your prompt :").strip() or "Fais-moi un état des lieux complet des tendances IA en Europe, structuré en sections."

def run_query(
    prompt: str = DEFAULT_PROMPT,
    model: str = "gemini-2.5-pro",
    temperature: float = 0.4,
    max_output_tokens: int = 1024,
    api_key: str | None = None,
):
    client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))  # :contentReference[oaicite:3]{index=3}

    # Nouveau tool 'google_search' (v2.x) — sinon utiliser google_search_retrieval (legacy) :contentReference[oaicite:4]{index=4}
    search_tool = types.Tool(google_search={})
    cfg = types.GenerateContentConfig(
        tools=[search_tool],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=cfg,
    )
    return resp.text

def main():
    ap = argparse.ArgumentParser(description="Gemini with Google Search grounding")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max_output_tokens", type=int, default=1024)
    ap.add_argument("--api_key", default=os.getenv("GEMINI_API_KEY"))
    args = ap.parse_args()
    print(run_query(**vars(args)))

if __name__ == "__main__":
    main()
