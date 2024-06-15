import argparse
import os
import re

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

DEVELOPER_KEY = os.environ["DEVELOPER_KEY"]
YOUTUBE_API_SERVICE = "youtube"
YOUTUBE_API_VERSION = "v3"


def extract_chapters_from_description(description):
    # 正規表現パターンの定義
    pattern = re.compile(r"(\d{2}:\d{2}:\d{2}) (.+)")

    # マッチ結果をリストに格納
    chapters = pattern.findall(description)

    return chapters


def youtube_search(args):
    youtube = build(
        YOUTUBE_API_SERVICE, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY
    )

    # Call the videos.list
    response = (
        youtube.videos()
        .list(
            id=args.video_id,
            part="snippet,contentDetails",
        )
        .execute()
    )

    for resp in response.get("items", []):
        title = resp["snippet"]["title"]
        channelTitle = resp["snippet"]["channelTitle"]
        description = resp["snippet"]["description"]
        chapters = extract_chapters_from_description(description)
        print("\n".join([f"{time} {text}" for time, text in chapters]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id", required=True, help="Youtube Video ID")
    args = parser.parse_args()

    try:
        youtube_search(args)
    except HttpError as e:
        print("An HTTP error %d occurred:\n%s" % (e.resp.status, e.content))
