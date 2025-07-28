import time
import argparse
from google import genai
from google.genai import types

from configs import GOOGLE_API_KEY, GOOGLE_MODELS
from cli_core import command, set_build_parser


@command('gemini-video', help='Generate video via Gemini API')
def gemini_veo3_video_main(args):
    main(args)

def main(args):
    prompt = args.prompt
    if not prompt:
        prompt = input("Enter a prompt: ").strip() or "A cat playing with a ball"

    model = args.model
    if args.image:
        model = "veo-2.0-generate-001"

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

@set_build_parser('gemini-video')
def build(p):
    p.add_argument("--prompt")
    p.add_argument("--negative-prompt")
    p.add_argument("--output")
    p.add_argument("--model", 
                        default="veo-3.0-generate-preview",
                        choices=["veo-2.0-generate-001"],
                        help="The model to use for video generation")
    p.add_argument("--api-key",
                        default=GOOGLE_API_KEY,
                        help="The API key to use for video generation")
    p.add_argument("--ar", default="16:9")
    p.add_argument("--image", default=None)
    p.add_argument("--n", default=1)
    args = p.parse_args()
    return args

if __name__ == "__main__":
    gemini_veo3_video_main(build(argparse.ArgumentParser()))