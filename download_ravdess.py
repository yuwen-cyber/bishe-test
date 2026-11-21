"""
多线程下载 RAVDESS zip，不用装任何工具。
用法：python download_ravdess.py
"""
import os
import threading
import requests
from pathlib import Path

# ===== 在这里改你的下载链接 =====
URL = "https://zenodo.org/records/1188976/files/Video_Speech_Actor_01.zip?download=1"
OUTPUT = r"D:\biyesheji\Video_Speech_Actor_01.zip"
THREADS = 16  # 16个线程同时下
# ================================

def get_file_size(url):
    r = requests.head(url, allow_redirects=True, timeout=30)
    return int(r.headers["Content-Length"])

def download_range(start, end, url, out_file, idx):
    headers = {"Range": f"bytes={start}-{end}"}
    r = requests.get(url, headers=headers, stream=True, timeout=60)
    with open(out_file, "r+b") as f:
        f.seek(start)
        for chunk in r.iter_content(chunk_size=1024*256):
            f.write(chunk)
    print(f"  线程 {idx:2d} 完成 ({end-start+1} bytes)")

def main():
    print(f"获取文件大小...")
    total = get_file_size(URL)
    print(f"总大小: {total/1024/1024:.1f} MB")

    # 预分配文件
    with open(OUTPUT, "wb") as f:
        f.truncate(total)

    # 分段
    chunk = total // THREADS
    threads = []
    for i in range(THREADS):
        start = i * chunk
        end = total - 1 if i == THREADS - 1 else (start + chunk - 1)
        t = threading.Thread(target=download_range, args=(start, end, URL, OUTPUT, i))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    size = os.path.getsize(OUTPUT)
    print(f"\n✅ 下载完成: {OUTPUT}")
    print(f"   文件大小: {size/1024/1024:.1f} MB")

if __name__ == "__main__":
    main()
