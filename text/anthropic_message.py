import argparse
import anthropic

from configs import ANTHROPIC_API_KEY, ANTHROPIC_MODELS

from cli_core import command, set_build_parser, positive_int
    

@command('anthropic-message', help='Generate text via Anthropic Chat API')
def anthropic_message_main(args):
    prompt = args.prompt
    if not prompt:
        prompt = input("Enter your prompt: ").strip()

    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=args.api_key,
    )

    message = client.messages.create(
        model=args.model,
        max_tokens=args.max_tokens,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=args.temperature
    )
    print(message.content)

    print(message.usage)



@set_build_parser('anthropic-message')
def build(p):
    p.add_argument('--api-key', aidefault=ANTHROPIC_API_KEY)
    p.add_argument('--max-tokens', type=positive_int, default=1024)
    p.add_argument('--model', default=ANTHROPIC_MODELS[0])
    p.add_argument('--prompt')
    args = p.parse_args()
    return args

if __name__ == "__main__":
    anthropic_message_main(build(argparse.ArgumentParser()))