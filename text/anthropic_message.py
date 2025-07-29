import argparse
import anthropic

from configs import ANTHROPIC_API_KEY, ANTHROPIC_MODELS

from cli_core import command, set_build_parser, positive_int
from argument_helpers import add_common_args, get_api_key_from_env

ANTHROPIC_API_KEY = get_api_key_from_env("ANTHROPIC_API_KEY")
ANTHROPIC_MODELS = ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"]

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
    # Ajouter les arguments communs
    add_common_args(p, ['api', 'model'])
    
    # Arguments spécifiques à Anthropic
    p.add_argument('--prompt', required=True, help='Text prompt')
    p.add_argument('--api-key', default=ANTHROPIC_API_KEY, help='Anthropic API key')
    p.add_argument('--model', choices=ANTHROPIC_MODELS, default=ANTHROPIC_MODELS[0])
    
    # Pas besoin de p.parse_args() - cli_core s'en charge
    args = p.parse_args()
    return args

if __name__ == "__main__":
    anthropic_message_main(build(argparse.ArgumentParser()))