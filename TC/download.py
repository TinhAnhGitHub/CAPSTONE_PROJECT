import yt_dlp
import argparse
import os
import json


def load_json(path):
    """Load or create the dataset JSON structure."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"video": [], "question": []}


def save_json(path, data):
    """Save dataset JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

import subprocess
def download_video(url: str, output_folder: str = "./videos", data : dict = None, id : int =None):
    """Download a YouTube video and record the URL in JSON."""
    os.makedirs(output_folder, exist_ok=True)
    ffmpeg_path = "./ffmpeg-8.0-essentials_build/bin"

    if id is None:
        id = len(data["video"])+1
    ydl_opts = {
        "outtmpl": f"{output_folder}/{id}_%(title)s.%(ext)s",
        "format": "bestvideo+bestaudio/best",
        "keepvideo": True,
        "ffmpeg_location":ffmpeg_path
        
    }

    try:
        if data and url not in data["video"]:
            data["video"].append(url)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        return True

        

    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return

def insert_questions(data: dict , question: str ):
    data["question"].append(question)
    print (f"✅ Question added: {question}. Total questions: {len(data['question'])}")
# 
def main():
    
    meta = load_json("./question.json")
    link = "https://youtu.be/Y3q6l8lzmOY"
    #res = download_video(link ,"./videos",meta, len(meta["video"])+1)

    while True:
        x = int(input("1. Add question     2. Download      3. exit: "))
        if x == 1:
            question = input("Enter your question: ")
            insert_questions(meta, question)
            save_json("./question.json", meta)
        elif x ==2:
            link =  input("Enter the YouTube video link: ")

            if link in meta["video"]:
                print(f"⚠️ Video already downloaded: {link}")
                print("Exiting... Stop being GAY brotha")
                return
            print(f"🎬 Downloading: {link }")
            download_video(link ,"./videos",meta)
            print(f"\n✅ Download complete. There are now {len(meta['video'])} videos in the dataset.")
            save_json("./question.json", meta)
        else:
            break
       


if __name__ == "__main__":
    main()
