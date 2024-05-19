import os

import cv2

CHAPTERS_24 = """
00:00:00 intro: Tokenization, GPT-2 paper, tokenization-related issues
00:05:50 tokenization by example in a Web UI (tiktokenizer)
00:14:56 strings in Python, Unicode code points
00:18:15 Unicode byte encodings, ASCII, UTF-8, UTF-16, UTF-32
00:22:47 daydreaming: deleting tokenization
00:23:50 Byte Pair Encoding (BPE) algorithm walkthrough
00:27:02 starting the implementation
00:28:35 counting consecutive pairs, finding most common pair
00:30:36 merging the most common pair
00:34:58 training the tokenizer: adding the while loop, compression ratio
00:39:20 tokenizer/LLM diagram: it is a completely separate stage
00:42:47 decoding tokens to strings
00:48:21 encoding strings to tokens
00:57:36 regex patterns to force splits across categories
01:11:38 tiktoken library intro, differences between GPT-2/GPT-4 regex
01:14:59 GPT-2 encoder.py released by OpenAI walkthrough
01:18:26 special tokens, tiktoken handling of, GPT-2/GPT-4 differences
01:25:28 minbpe exercise time! write your own GPT-4 tokenizer
01:28:42 sentencepiece library intro, used to train Llama 2 vocabulary
01:43:27 how to set vocabulary set? revisiting gpt.py transformer
01:48:11 training new tokens, example of prompt compression
01:49:58 multimodal [image, video, audio] tokenization with vector quantization
01:51:41 revisiting and explaining the quirks of LLM tokenization
02:10:20 final recommendations
02:12:50 ??? :)
"""


def chapters_to_list(chapters):
    chapters_list = chapters.strip().split("\n")
    chapters_dict_list = []
    for chapter in chapters_list:
        time_str, topic = chapter.split(" ", 1)
        hours, minutes, seconds = map(int, time_str.split(":"))
        total_seconds = hours * 3600 + minutes * 60 + seconds
        chapters_dict_list.append({"timestamp": total_seconds, "topic": topic})
    return chapters_dict_list


def get_text_chapter(transcript, chapter_start_time, chapter_end_time, output_dir):
    text_chapter = ""
    for i in range(len(transcript)):
        transcript_i = transcript[i]

    if (
        int(transcript_i["start"]) >= chapter_start_time
        and int(transcript_i["start"]) <= chapter_end_time
    ):
        text_chapter += transcript_i["text"].replace("\n", "").strip() + " "

    transcript_file = os.path.join(output_dir, "transcript.txt")
    with open(transcript_file, "w") as f:
        f.write(text_chapter)


def get_frames_chapter(
    video_path,
    chapter_start_time,
    chapter_end_time,
    output_dir,
    timestamps_screenshots=None,
):
    if timestamps_screenshots is None:
        sceenshot_interval = int((chapter_end_time - chapter_start_time) / 10)
        if sceenshot_interval < 60:
            sceenshot_interval = 60
        timestamps_screenshots = list(
            range(chapter_start_time, chapter_end_time, sceenshot_interval)
        )
    else:
        timestamps_screenshots = [int(ts) for ts in timestamps_screenshots]

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    for timestamp in timestamps_screenshots:
        index = int(timestamp * fps)
        video.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = video.read()
        if success:
            timestamp_str = "{:05d}".format(timestamp)
            output_path = f"{output_dir}/{timestamp_str}.jpg"
            cv2.imwrite(output_path, frame)
    video.release()


def chop_up_in_chapters(
    chapters_list,
    video_path,
    transcript,
    chapters_dir,
    timestamps_screenshots_list_seconds=None,
):
    n_chapters = len(chapters_list) - 1
    print(f"Number of chunks: {n_chapters}")

    for current_chapter in range(n_chapters):
        output_dir = os.path.join(chapters_dir, str(current_chapter))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        current_chunk_start_time = chapters_list[current_chapter]["timestamp"]
        current_chunk_end_time = chapters_list[current_chapter + 1]["timestamp"] - 1
        print(
            f"Chapter {current_chapter}; Start: {current_chunk_start_time}, End: {current_chunk_end_time}"
        )

        get_text_chapter(
            transcript, current_chunk_start_time, current_chunk_end_time, output_dir
        )

        if timestamps_screenshots_list_seconds is not None:
            get_frames_chapter(
                video_path,
                current_chunk_start_time,
                current_chunk_end_time,
                output_dir,
                timestamps_screenshots_list_seconds[current_chapter],
            )
        else:
            get_frames_chapter(
                video_path, current_chunk_start_time, current_chunk_end_time, output_dir
            )


if __name__ == "__main__":
    chapters_list = chapters_to_list(CHAPTERS_24)
    print(f"chapters_list length: {len(chapters_list)}")
    print(f"last chapter: {chapters_list[-1]}")
