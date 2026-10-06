import os
import sys
import subprocess
import time
from pathlib import Path

URL = "https://huggingface.co/google/siglip2-so400m-patch14-384/resolve/main/model.safetensors"
OUT_DIR = Path(r"C:\Users\AI Fiham\.cache\huggingface\hub\models--google--siglip2-so400m-patch14-384\snapshots\e8e487298228002f3d8a82e0cd5c8ea9c567f57f")
OUT_FILE = OUT_DIR / "model.safetensors"
TOTAL_SIZE = 4544143072
NUM_CHUNKS = 6

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_FILE.exists() and OUT_FILE.stat().st_size == TOTAL_SIZE:
        print(f"File already complete: {OUT_FILE} ({TOTAL_SIZE} bytes)")
        return

    chunk_size = (TOTAL_SIZE + NUM_CHUNKS - 1) // NUM_CHUNKS
    procs = []
    chunk_files = []

    print(f"Starting {NUM_CHUNKS} parallel curl download streams...")
    t0 = time.time()

    for i in range(NUM_CHUNKS):
        start = i * chunk_size
        end = min(TOTAL_SIZE - 1, (i + 1) * chunk_size - 1)
        part_file = OUT_DIR / f"chunk_{i:02d}.part"
        chunk_files.append((part_file, end - start + 1))

        # Check existing size for resume
        curr_size = part_file.stat().st_size if part_file.exists() else 0
        if curr_size == (end - start + 1):
            print(f"Chunk {i:02d} already complete.")
            continue

        cmd = [
            "curl.exe", "-s", "-L",
            "-r", f"{start + curr_size}-{end}",
            "-o", str(part_file),
            URL
        ]
        # Append if resuming
        if curr_size > 0:
            cmd[cmd.index("-o")] = "-a"  # curl doesn't use -a for files, use redirect or standard write

        # To be clean, if not complete, download with -r start-end
        cmd = [
            "curl.exe", "-s", "-L",
            "-r", f"{start}-{end}",
            "-o", str(part_file),
            URL
        ]
        p = subprocess.Popen(cmd)
        procs.append((i, p, part_file))

    print(f"Spawned {len(procs)} curl download processes.")

    # Monitor progress
    while any(p.poll() is None for _, p, _ in procs):
        time.sleep(5)
        total_dl = sum(pf.stat().st_size for pf, _ in chunk_files if pf.exists())
        pct = total_dl / TOTAL_SIZE * 100
        elapsed = time.time() - t0
        speed = (total_dl / (1024**2)) / max(0.1, elapsed)
        print(f"Progress: {total_dl / (1024**2):.1f} MB / {TOTAL_SIZE / (1024**2):.1f} MB ({pct:.1f}%) | Speed: {speed:.1f} MB/s | Elapsed: {elapsed:.0f}s", flush=True)

    # Check return codes
    for i, p, pf in procs:
        if p.returncode != 0:
            print(f"ERROR: Chunk {i} curl failed with returncode {p.returncode}")
            sys.exit(1)

    print("All chunks downloaded successfully! Concatenating chunks into model.safetensors...", flush=True)
    with open(OUT_FILE, "wb") as outfile:
        for pf, expected in chunk_files:
            actual = pf.stat().st_size
            if actual != expected:
                print(f"ERROR: Chunk {pf.name} size {actual} != expected {expected}")
                sys.exit(1)
            with open(pf, "rb") as infile:
                while True:
                    buf = infile.read(32 * 1024 * 1024)  # 32MB buffer
                    if not buf:
                        break
                    outfile.write(buf)

    final_size = OUT_FILE.stat().st_size
    print(f"Final file size: {final_size} bytes (Expected: {TOTAL_SIZE})", flush=True)
    if final_size == TOTAL_SIZE:
        print("Verification SUCCESS! Cleaning up temporary chunk files...", flush=True)
        for pf, _ in chunk_files:
            try:
                pf.unlink()
            except Exception:
                pass
        print(f"Download and assembly COMPLETE in {time.time() - t0:.1f}s!", flush=True)
    else:
        print("ERROR: Final file size mismatch!", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
