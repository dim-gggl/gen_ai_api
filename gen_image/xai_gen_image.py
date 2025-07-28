import argparse

import openai
from cli_core import command, set_build_parser

from configs import XAI_MODELS, XAI_API_KEY

@command('xai-image', help='Generate image via xAI Grok API')
def xai_image_main(args):
    main(args)

def main(args):
    prompt = args.prompt
    if not prompt:
        prompt = input("Enter a prompt: ").strip() or "Mylène Farmer version canine"

    client = openai.OpenAI(api_key=XAI_API_KEY)

    response = client.images.generate(
        model=args.model,
        prompt=prompt,
        image_format=args.format,
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print(response.data[0].url)

@set_build_parser('xai-image')
def build(p):
    p.add_argument("--prompt",
                        help="Prompt for the image")
    p.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of images to generate")
    p.add_argument("--model",
                        choices=[mod for mod in XAI_MODELS if "image" in mod],
                        default=XAI_MODELS[-1])
    p.add_argument("--format",
                        default="url",
                        nargs="image_format")

    args = p.parse_args()
    return args

if __name__ == "__main__":
    xai_image_main(build(argparse.ArgumentParser()))