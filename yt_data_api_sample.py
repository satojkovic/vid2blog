import argparse
import os

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

DEVELOPER_KEY = os.environ["DEVELOPER_KEY"]
YOUTUBE_API_SERVICE = "youtube"
YOUTUBE_API_VERSION = "v3"


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
        print(resp["snippet"]["title"])
        print(resp["snippet"]["channelTitle"])
        print(resp["snippet"]["description"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id", required=True, help="Youtube Video ID")
    args = parser.parse_args()

    try:
        youtube_search(args)
    except HttpError as e:
        print("An HTTP error %d occurred:\n%s" % (e.resp.status, e.content))
