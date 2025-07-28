import argparse
from xai_sdk import Client

from p_ai.configs import XAI_MODELS, XAI_API_KEY

client = Client(api_key=XAI_API_KEY)

parser = argparse.ArgumentParser(
    description="Generate image via xAI Grok API"
)

parser.add_argument("--prompt",
                    default=input("Enter a prompt: ").strip() or "Mylène Farmer version canine",
                    help="Prompt for the image")
parser.add_argument("--n",
                    type=int,
                    default=1,
                    help="Number of images to generate")
parser.add_argument("--model",
                    choices=[mod for mod in XAI_MODELS if "image" in mod],
                    default=XAI_MODELS[-1])
parser.add_argument("--format",
                    default="url",
                    nargs="image_format")

args = parser.parse_args()

response = client.image.sample(
    model=args.model,
    prompt=args.prompt,
    image_format=args.format,
    n=args.n,
    temperature=args.temperature,
    top_p=args.top_p,
)

print(response.url)