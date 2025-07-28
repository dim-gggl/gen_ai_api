import os
from openai import OpenAI
import base64
import argparse

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

parser = argparse.ArgumentParser(
    description="Generate image via OpenAI API"
)

parser.add_argument("--prompt",
                    default=input("Enter a prompt: ").strip() or "Cat as a president",
                    help="Prompt for the image")
parser.add_argument("--n",
                    type=int,
                    default=1,
                    help="Number of images to generate")
parser.add_argument("--model",
                    choices=["gpt-image-1", "dall-e-3"],
                    default="gpt-image-1",
                    help="Model to use")

args = parser.parse_args()

img = client.images.generate(
    model=args.model,
    prompt=args.prompt,
    n=args.n,
    size="1024x1024"
)

image_base64 = img.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

# Save the image to a file
with open("otter.png", "wb") as f:
    f.write(image_bytes)


