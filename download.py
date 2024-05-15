import os

import pytube


def download_video(video_id, output_path):
    youtube = pytube.YouTube(f"https://www.youtube.com/watch?v={video_id}")
    stream = youtube.streams.get_highest_resolution()
    video_path = stream.download(output_path=output_path, filename=video_id + ".mp4")
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
