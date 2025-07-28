import anthropic

from configs import ANTHROPIC_API_KEY, ANTHROPIC_MODELS

from cli_core import command, set_build_parser, positive_int
    

@command('anthropic', help='Generate text via Anthropic Chat API')
def main(args):
    # print("Starting Anthropic Message Generator")
    # parser = argparse.ArgumentParser(
    #         description="Generate text via Anthropic Chat Completions API"
    #     )
    # parser.add_argument("--api-key",
    #                     default=ANTHROPIC_API_KEY,
    #                     help="Anthropic API key (env: ANTHROPIC_API_KEY)")
    # parser.add_argument("--api-base",
    #                     default=os.getenv("ANTHROPIC_BASE_URL"),
    #                     help="Custom API base URL if using a proxy or enterprise endpoint")
    # parser.add_argument("--timeout",
    #                     type=float,
    #                     default=600.0,
    #                     help="Timeout for requests, in seconds (default 10 minutes)")
    # parser.add_argument("--max-retries",
    #                     type=int,
    #                     default=2,
    #                     help="Number of automatic retries for transient errors")
    # parser.add_argument("--model",
    #                     default=ANTHROPIC_MODELS[0],
    #                     help="Chat model to use")
    # parser.add_argument("--prompt",
    #                     default=input("Enter your prompt: ").strip(),
    #                     help="Prompt to send as the user message")
    # parser.add_argument("--max-tokens",
    #                     type=int,
    #                     default=1024,
    #                     help="Maximum number of tokens to generate")
    # parser.add_argument("--thinking",
    #                     type=bool,
    #                     default=True,
    #                     help="Whether to enable thinking")
    # parser.add_argument("--tool-choice",
    #                     type=str,
    #                     default="auto",
    #                     help="Tool choice")
    # parser.add_argument("--tools",
    #                     type=list,
    #                     default=[{}],
    #                     help="Tools to use")
    # parser.add_argument("--system",
    #                     type=str,
    #                     default="You are a helpful assistant that can answer questions and help with tasks.",
    #                     help="System prompt")
    # parser.add_argument("--metadata",
    #                     type=dict,
    #                     default={},
    #                     help="Metadata")
    # parser.add_argument("--container",
    #                     type=dict,
    #                     default={},
    #                     help="Container")
    # parser.add_argument("--service-tier",
    #                     type=str,
    #                     default="",
    #                     help="Service tier")
    # parser.add_argument("--stop-sequences",
    #                     type=list,
    #                     default=[],
    #                     help="Stop sequences")
    # parser.add_argument("--top-k",
    #                     type=int,
    #                     default=0,
    #                     help="Top K")
    # parser.add_argument("--top-p",
    #                     type=float,
    #                     default=0.0,
    #                     help="Top P")
    # parser.add_argument("--temperature",
    #                     type=float,
    #                     default=0.0,
    #                     help="Temperature")

    # args = parser.parse_args()

    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=args.api_key,
    )

    message = client.messages.create(
        model=args.model,
        max_tokens=args.max_tokens,
        messages=[
            {"role": "user", "content": args.prompt}
        ],
        temperature=args.temperature
    )
    print(message.content)

    print(message.usage)



@set_build_parser('anthropic')
def build(p):
    p.add_argument('--api-key', aidefault=ANTHROPIC_API_KEY)
    p.add_argument('--max-tokens', type=positive_int, default=1024)
    p.add_argument('--prompt')
    p.add_argument('--model', default=ANTHROPIC_MODELS[0])

if __name__ == "__main__":
    args = build(argparse.ArgumentParser())
    main(args)