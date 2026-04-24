"""
ISIA Food-500 Downloader — with resume + auto-retry
"""

import os
import sys
import time
import requests
from pathlib import Path

BASE_URL   = "http://123.57.42.89/Dataset_ict/ISIA_Food500_Dir/dataset/"
DEST_DIR   = Path("/aux/s22imc10262/NLP_hackathon_data/full_data_isia-food-500")
CHUNK_SIZE = 4 * 1024 * 1024   # 4 MB
MAX_RETRY  = 99999              # effectively retry forever
TIMEOUT    = 60                 # seconds per read

FILES = [
    "ISIA_Food500.z01", "ISIA_Food500.z02", "ISIA_Food500.z03",
    "ISIA_Food500.z04", "ISIA_Food500.z05", "ISIA_Food500.z06",
    "ISIA_Food500.z07", "ISIA_Food500.z08", "ISIA_Food500.z09",
    "ISIA_Food500.z10", "ISIA_Food500.zip", "metadata_ISIAFood_500.zip",
]


def fmt(b):
    for u in ["B", "KB", "MB", "GB"]:
        if b < 1024: return f"{b:.1f} {u}"
        b /= 1024
    return f"{b:.1f} TB"


def get_remote_size(url: str) -> int:
    """HEAD request to get Content-Length."""
    try:
        r = requests.head(url, timeout=30)
        return int(r.headers.get("content-length", 0))
    except Exception:
        return 0


def download_file(filename: str, dest_dir: Path):
    url  = BASE_URL + filename
    dest = dest_dir / filename
    tmp  = dest_dir / (filename + ".part")

    remote_size = get_remote_size(url)

    # Already fully downloaded?
    if dest.exists():
        local = dest.stat().st_size
        if remote_size and local >= remote_size:
            print(f"  [SKIP] {filename} ({fmt(local)})")
            return
        else:
            # Rename back to .part so we resume it
            dest.rename(tmp)
            print(f"  [INCOMPLETE] {filename} — resuming from {fmt(tmp.stat().st_size)}")

    attempt = 0
    while attempt < MAX_RETRY:
        attempt += 1
        resume_pos = tmp.stat().st_size if tmp.exists() else 0

        if resume_pos and remote_size and resume_pos >= remote_size:
            print(f"  [DONE already] {filename}")
            tmp.rename(dest)
            return

        headers = {"Range": f"bytes={resume_pos}-"} if resume_pos else {}
        label   = f"attempt {attempt}" if attempt > 1 else "downloading"
        print(f"  [{label}] {filename}  resume={fmt(resume_pos)}", flush=True)

        try:
            start = time.time()
            with requests.get(url, headers=headers, stream=True, timeout=TIMEOUT) as r:
                # 206 = partial content (resume), 200 = full (server ignored Range)
                if r.status_code == 200 and resume_pos:
                    print("    Server ignored Range header — restarting from 0")
                    resume_pos = 0
                    tmp.unlink(missing_ok=True)

                r.raise_for_status()
                total     = resume_pos + int(r.headers.get("content-length", 0))
                mode      = "ab" if resume_pos else "wb"
                downloaded = resume_pos

                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            elapsed = time.time() - start
                            speed   = (downloaded - resume_pos) / elapsed if elapsed else 0
                            pct     = downloaded / total * 100 if total else 0
                            print(
                                f"    {fmt(downloaded)} / {fmt(total)}"
                                f"  ({pct:.1f}%)  {fmt(speed)}/s    ",
                                end="\r", flush=True,
                            )

            # Verify size
            final_size = tmp.stat().st_size
            if remote_size and final_size < remote_size:
                print(f"\n    [WARN] Got {fmt(final_size)}, expected {fmt(remote_size)} — retrying")
                time.sleep(5)
                continue

            tmp.rename(dest)
            elapsed = time.time() - start
            print(f"\n    Saved {fmt(dest.stat().st_size)} in {elapsed:.0f}s")
            return

        except Exception as e:
            wait = min(30 * attempt, 300)   # back-off up to 5 min
            print(f"\n    [ERROR] {e}")
            print(f"    Retrying in {wait}s (attempt {attempt}/{MAX_RETRY}) …")
            time.sleep(wait)

    print(f"[FATAL] Could not download {filename} after {MAX_RETRY} attempts.")
    sys.exit(1)


def extract(dest_dir: Path):
    zip_path    = dest_dir / "ISIA_Food500.zip"
    extract_dir = dest_dir / "images"
    extract_dir.mkdir(exist_ok=True)
    print(f"\nExtracting to {extract_dir} …")
    ret = os.system(f'unzip -o "{zip_path}" -d "{extract_dir}"')
    if ret != 0:
        print(f"[ERROR] unzip failed. Run manually:\n  cd {dest_dir} && unzip ISIA_Food500.zip -d images/")
        sys.exit(1)
    print("Extracting metadata …")
    os.system(f'unzip -o "{dest_dir / "metadata_ISIAFood_500.zip"}" -d "{dest_dir / "metadata"}"')
    print("Done.")


if __name__ == "__main__":
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Destination : {DEST_DIR}")
    print(f"Files       : {len(FILES)}  (~43 GB total)\n")

    for fname in FILES:
        download_file(fname, DEST_DIR)

    print("\nAll files downloaded.")
    if input("\nExtract now? [y/N] ").strip().lower() == "y":
        extract(DEST_DIR)
    else:
        print(f"\nTo extract later:\n  cd {DEST_DIR} && unzip ISIA_Food500.zip -d images/")