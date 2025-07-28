import os
from openai import OpenAI
import base64
import argparse

from configs import OPENAI_API_KEY, OPENAI_MODELS
from cli_core import command, set_build_parser


@command('openai-image', help='Generate image via OpenAI API')
def openai_image_main(args):
    main(args)


def main(args):
    prompt = args.prompt
    if not prompt:
        prompt = input("Enter a prompt: ").strip() or "Cat as a president"

    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    img = client.images.generate(
        model=args.model,
        prompt=prompt,
        n=args.n,
        size="1024x1024"
    )

    img = client.images.generate(
        model=args.model,
        prompt=prompt,
        n=args.n,
        size="1024x1024"
    )

    image_base64 = img.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)

    # Save the image to a file
    with open("otter.png", "wb") as f:
        f.write(image_bytes)


@set_build_parser('openai-image')
def build(p):
    p.add_argument("--model",
                    choices=["gpt-image-1", "dall-e-3"],
                    default="gpt-image-1",
                    help="Model to use")
    p.add_argument("--prompt",
                    help="Prompt for the image")
    p.add_argument("--n",
                    type=int,
                    default=1,
                    help="Number of images to generate")

    args = p.parse_args()
    return args

if __name__ == "__main__":
    openai_image_main(build(argparse.ArgumentParser()))