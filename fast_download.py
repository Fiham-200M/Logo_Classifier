import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "https://huggingface.co/google/siglip2-so400m-patch14-384/resolve/main/model.safetensors"
OUT_PATH = r"C:\Users\AI Fiham\.cache\huggingface\hub\models--google--siglip2-so400m-patch14-384\snapshots\e8e487298228002f3d8a82e0cd5c8ea9c567f57f\model.safetensors"
TOTAL_SIZE = 4544143072
NUM_WORKERS = 16

def download_chunk(worker_id, start, end, filename):
    headers = {"Range": f"bytes={start}-{end}"}
    req = urllib.request.Request(URL, headers=headers)
    retries = 3
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
                expected = end - start + 1
                if len(data) != expected:
                    raise IOError(f"Worker {worker_id}: read {len(data)} != expected {expected}")
                with open(filename, "r+b") as f:
                    f.seek(start)
                    f.write(data)
                return worker_id, len(data)
        except Exception as e:
            if attempt == retries - 1:
                raise e
            time.sleep(1 + attempt)

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    
    # Check if already complete
    if os.path.exists(OUT_PATH) and os.path.getsize(OUT_PATH) == TOTAL_SIZE:
        print(f"File already complete: {OUT_PATH} ({TOTAL_SIZE} bytes)")
        return

    # Pre-allocate sparse file
    print(f"Allocating {TOTAL_SIZE / (1024**3):.2f} GB for {OUT_PATH}...")
    with open(OUT_PATH, "wb") as f:
        f.seek(TOTAL_SIZE - 1)
        f.write(b"\0")

    chunk_size = (TOTAL_SIZE + NUM_WORKERS - 1) // NUM_WORKERS
    tasks = []
    for i in range(NUM_WORKERS):
        start = i * chunk_size
        end = min(TOTAL_SIZE - 1, (i + 1) * chunk_size - 1)
        if start <= end:
            tasks.append((i, start, end))

    print(f"Starting multi-threaded download ({NUM_WORKERS} workers)...")
    t0 = time.time()
    downloaded = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(download_chunk, t[0], t[1], t[2], OUT_PATH): t for t in tasks}
        for future in as_completed(futures):
            worker_id, nbytes = future.result()
            downloaded += nbytes
            pct = downloaded / TOTAL_SIZE * 100
            elapsed = time.time() - t0
            speed_mb = (downloaded / (1024**2)) / max(0.1, elapsed)
            print(f"Worker {worker_id:02d} done ({nbytes/(1024**2):.1f} MB) | Total: {pct:.1f}% | Avg Speed: {speed_mb:.1f} MB/s | Elapsed: {elapsed:.1f}s")

    print(f"Download COMPLETE! Total time: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
