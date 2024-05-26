import base64
import glob
import os
import re
from pathlib import Path

import openai
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
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64, {base64.b64encode(open(screenshot, 'rb').read()).decode('utf-8')}",
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
            "role": "system",
            "content": prompt_instructions,
        },
        {
            "role": "user",
            "content": screenshots_as_messagges
            + [{"type": "text", "text": f"<transcript>\n{transcript}\n</transcript>"}],
        },
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

    video_path = os.path.join(DATA_DIR, f"{VIDEO_ID}.mp4")
    if not os.path.exists(video_path):
        video_path = download_video(VIDEO_ID, DATA_DIR)

    # Transcript
    transcript = YouTubeTranscriptApi.get_transcript(VIDEO_ID)
    print(f"transcript length: {len(transcript)}")
    print(f"{transcript[0]}")

    # Chop up to chpter
    chapters_list = chapters_to_list(CHAPTERS_24)
    chop_up_in_chapters(chapters_list, video_path, transcript, CHAPTERS_DIR)

    # OpenAI client
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    for chapter in range(len(chapters_list) - 1):
        print(f"Processing chunk {chapter}")

        # Generate the prompt for the current chapter
        prompt_generete_markdown = get_prompt_as_messages(chapter, CHAPTERS_DIR)

        # Create a message by invoking Claude with the prompt
        message = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            max_tokens=4000,
            messages=prompt_generete_markdown,
        )

        # Extract the generated markdown content from the response
        answer = message.choices[0].message.content
        markdown = "# " + answer

        # Define the path for the markdown file corresponding to the current chapter
        markdown_file = os.path.join(CHAPTERS_DIR, str(chapter), "markdown.md")

        # Write the generated markdown content to the file
        with open(markdown_file, "w") as f:
            f.write(markdown)

    #
    # Post processing
    #
    merged_markdown = ""

    for chapter in range(len(chapters_list) - 1):
        markdown_file = os.path.join(CHAPTERS_DIR, str(chapter), "markdown.md")
        with open(markdown_file, "r") as f:
            markdown = f.readlines()
        # Let us add, for each chapter title, a hyperlink to the video at the right timestamp
        url_chapter = f"https://www.youtube.com/watch?v={VIDEO_ID}&t={chapters_list[chapter]['timestamp']}s"
        markdown[0] = f"# [{chapter + 1}) {markdown[0][2:].strip()}]({url_chapter})"
        markdown = "\n".join(markdown)

        merged_markdown += "\n" + markdown

    # Find all <img> tags with timestamps in the src attribute, so we can add a hyperlink to the video at the right timestamp
    timestamps_screenshots = re.findall(r'<img src="(\d+)\.jpg"[^>]*>', merged_markdown)
    timestamps_screenshots = [timestamp for timestamp in timestamps_screenshots]

    # Add a hyperlink to the video at the right timestamp for each image
    for timestamp in timestamps_screenshots:
        video_link = f'<a href="https://www.youtube.com/watch?v={VIDEO_ID}&t={int(timestamp)}s">Link to video</a>'
        merged_markdown = merged_markdown.replace(
            f'<img src="{timestamp}.jpg"/>',
            f'<img src="{timestamp}.jpg"/>\n\n{video_link}',
        )

    # Get frames based on screenshots effectively selected in the merged markdown and save in merge folder
    get_frames_chapter(
        video_path, None, None, MERGE_DIR, timestamps_screenshots=timestamps_screenshots
    )

    # Save the merged markdown to a markdown blogpost.md file
    markdown_file = os.path.join(MERGE_DIR, "blogpost.md")
    with open(markdown_file, "w") as f:
        f.write(merged_markdown)
