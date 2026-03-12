import os
import json
import subprocess
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from audiocaps_download import Downloader
# ================== 配置 ==================
SAVE_DIR = "videos"
JSONL_PATH = "data.jsonl"
NUM_WORKERS = 8
TIMEOUT = 300

os.makedirs(SAVE_DIR, exist_ok=True)

# ================== 下载函数 ==================
def download_video(sample, idx):
    url = sample["url"]
    text = sample["text"]

    # 解析时间戳
    ts = sample.get("timestamp", None)
    if ts:
        start, end = json.loads(ts)
    else:
        start, end = None, None

    video_path = os.path.join(SAVE_DIR, f"{idx:08d}.mp4")

    # 已存在直接返回
    if os.path.exists(video_path) and os.path.getsize(video_path) > 1024:
        return {"video": video_path, "text": text}

    try:
        # yt-dlp 下载 + 裁剪
        cmd = [
            "yt-dlp",
            "--cookies", "cookies.txt",
            "-o", video_path,
            url
        ]

        # if start is not None:
        #     cmd += ["--download-sections", f"*{start}-{end}"]

        cmd.append(url)

        subprocess.run(cmd, timeout=TIMEOUT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if os.path.exists(video_path) and os.path.getsize(video_path) > 1024:
            return {"video": video_path, "text": text}

    except Exception as e:
        print(f"[FAIL] {url} -> {e}")

    return None

# ================== 主流程 ==================
def main():
    ds = load_dataset("DropletX/DropletVideo-10M")['train']
    ds = ds.select(range(100))
    results = []
    futures = []

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for i, sample in enumerate(ds):
            futures.append(executor.submit(download_video, sample, i))

        for future in tqdm(as_completed(futures), total=len(futures)):
            res = future.result()
            if res is not None:
                results.append(res)

    # 写 jsonl
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} samples to {JSONL_PATH}")



def audio():
    
    d = Downloader(root_path='audiocaps/', n_jobs=16)
    d.download(format='wav')


if __name__ == "__main__":
    audio()
