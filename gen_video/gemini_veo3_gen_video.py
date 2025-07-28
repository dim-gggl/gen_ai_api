import time
import argparse
from google import genai
from google.genai import types

from configs import GOOGLE_API_KEY, GOOGLE_MODELS


def main():

    parser = argparse.ArgumentParser(
        description="A video generator using Gemini API"
    )
    parser.add_argument("--prompt")
    parser.add_argument("--negative-prompt")
    parser.add_argument("--output")
    parser.add_argument("--model", 
                        default="veo-3.0-generate-preview",
                        choices=["veo-2.0-generate-001"],
                        help="The model to use for video generation")
    parser.add_argument("--api-key",
                        default=GOOGLE_API_KEY,
                        help="The API key to use for video generation")
    parser.add_argument("--ar", default="16:9")
    parser.add_argument("--image", default=None)
    parser.add_argument("--n", default=1)
    args = parser.parse_args()

    model = args.model
    if args.image:
        model = "veo-2.0-generate-001"
    prompt = args.prompt
    if not prompt:
        prompt = input("Your prompt : ").strip()

    client = genai.Client(api_key=args.api_key)

    operation = client.models.generate_videos(
        model=model,
        prompt=prompt,
        image=args.image or None,
        config=types.GenerateVideosConfig(
            negative_prompt=args.negative_prompt,
            aspect_ratio=args.ar,  # 16:9 is the only supported for Veo 3
            number_of_videos=args.n, # 1 video generated per request
        ),
    )

    # Waiting for the video(s) to be generated
    while not operation.done:
        time.sleep(2)
        print(f"\t\tça génère", end="")
        time.sleep(1)
        print(".", end="")
        time.sleep(1)
        print(".", end="")
        time.sleep(1)
        print(".")
        time.sleep(15)
        operation = client.operations.get(operation)


    generated_video = operation.result.generated_videos[0]
    client.files.download(file=generated_video.video)
    file_name = args.output.split("/")[-1] if args.output else f"{str(hash(str(operation)))[:5]}_veo3_video.mp4"
    output = args.output or f"gen_video/output/veo3/{file_name}"
    generated_video.video.save(output)

if __name__ == "__main__":
    main()