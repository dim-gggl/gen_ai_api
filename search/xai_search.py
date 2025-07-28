#!/usr/bin/env python3
"""
Query Grok 4 with Live Search enabled and get a long, structured answer.
"""
import argparse, asyncio, os, textwrap
import xai_sdk

DEFAULT_PROMPT = input("Enter your prompt :").strip() or "Fais-moi un état des lieux complet des tendances IA en Europe, structuré en sections."

async def run_query(
    prompt: str = DEFAULT_PROMPT,
    model: str = "grok-4-online",
    max_tokens: int = 1024,
    temperature: float = 0.3,
    search_mode: str = "on",           # off | auto | on
    api_key: str | None = None,
    api_host: str | None = None,
) -> str:
    client = xai_sdk.AsyncClient(
        api_key=api_key or os.getenv("XAI_API_KEY"),
        api_host=api_host or "api.x.ai",
    )  # :contentReference[oaicite:0]{index=0}

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        search_parameters={"mode": search_mode},  # Live Search on :contentReference[oaicite:1]{index=1}
    )
    return response.choices[0].message.content

async def _amain():
    ap = argparse.ArgumentParser(description="Grok 4 + Live Search")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--model", default="grok-4-online")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--search_mode", choices=["off", "auto", "on"], default="on")
    ap.add_argument("--api_key", default=os.getenv("XAI_API_KEY"))
    ap.add_argument("--api_host", default="api.x.ai")
    args = ap.parse_args()
    print(await run_query(**vars(args)))

if __name__ == "__main__":
    asyncio.run(_amain())
