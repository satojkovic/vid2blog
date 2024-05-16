import os

import pytube
from tqdm import tqdm


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
