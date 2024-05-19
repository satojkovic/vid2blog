import base64
import glob
import os
from pathlib import Path

import anthropic
import pytube
from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi

from config import *
from preproc_func import *


def progress_function(stream, chunk, bytes_remaining):
    progress_bar.update(len(chunk))


def download_video(video_id, output_path):
    youtube = pytube.YouTube(
        f"https://www.youtube.com/watch?v={video_id}",
        on_progress_callback=progress_function,
    )
    stream = youtube.streams.get_highest_resolution()
    total_size = stream.filesize
    global progress_bar
    progress_bar = tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading")
    video_path = stream.download(output_path=output_path, filename=video_id + ".mp4")
    progress_bar.close()
    return video_path


def get_screenshots_as_messages(screenshots):
    """
    The function iterates over all screenshots in order to describe each of them with two messages:
    - a text message that specifies the timestamp for the screenshot
    - an image message containing its base64-encoded representation
    """
    screenshots_as_messages = []
    for screenshot in screenshots:
        screenshots_as_messages.extend(
            [
                {
                    "type": "text",
                    "text": f"The timestamp for the following image is {Path(screenshot).stem}",
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.b64encode(open(screenshot, "rb").read()).decode(
                            "utf-8"
                        ),
                    },
                },
            ]
        )

    return screenshots_as_messages


def get_prompt_as_messages(chapter_id, chapters_dir):
    folder_path = os.path.join(chapters_dir, str(chapter_id))
    with open(os.path.join(folder_path, "transcript.txt"), "r") as f:
        transcript = f.read()

    screenshots = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
    screenshots_as_messagges = get_screenshots_as_messages(screenshots)
    prompt_as_messages = [
        {
            "role": "user",
            "content": screenshots_as_messagges
            + [{"type": "text", "text": f"<transcript>\n{transcript}\n</transcript>"}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "#"}]},
    ]

    return prompt_as_messages


if __name__ == "__main__":
    # Andrej Karpathy : Let's build the GPT Tokenizer - https://www.youtube.com/watch?v=zduSFxRajkE
    VIDEO_ID = "zduSFxRajkE"

    DATA_DIR = VIDEO_ID
    CHAPTERS_DIR = os.path.join(DATA_DIR, "chapters")
    MERGE_DIR = os.path.join(DATA_DIR, "final_output")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if not os.path.exists(CHAPTERS_DIR):
        os.makedirs(CHAPTERS_DIR)

    if not os.path.exists(MERGE_DIR):
        os.makedirs(MERGE_DIR)

    video_path = download_video(VIDEO_ID, DATA_DIR)

    # Transcript
    transcirpt = YouTubeTranscriptApi.get_transcript(VIDEO_ID)
    print(f"transcript length: {len(transcirpt)}")
    print(f"{transcirpt[0]}")

    # Chop up to chpter
    chapters_list = chapters_to_list(CHAPTERS_24)
    chop_up_in_chapters(chapters_list, video_path, transcirpt, CHAPTERS_DIR)

    # Call Claude API
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    for chapter in range(len(chapters_list) - 1):
        print(f"Processing chunk {chapter}")

        # Generate the prompt for the current chapter
        prompt_generete_markdown = get_prompt_as_messages(chapter, CHAPTERS_DIR)

        # Create a message by invoking Claude with the prompt
        message = client.messages.create(
            model="claude-3-opus-20240229",
            system="You are an expert at writing markdown blog post.",
            temperature=0,
            max_tokens=4000,
            messages=prompt_generete_markdown,
        )

        # Extract the generated markdown content from the response
        answer = message.content[0].text
        markdown = "#" + answer

        # Define the path for the markdown file corresponding to the current chapter
        markdown_file = os.path.join(CHAPTERS_DIR, str(chapter), "markdown.md")

        # Write the generated markdown content to the file
        with open(markdown_file, "w") as f:
            f.write(markdown)
