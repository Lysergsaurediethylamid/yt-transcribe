"""
功能：
  - 下载 YouTube 视频音频（多线程加速）
  - 调用 OpenAI gpt-4o-mini-transcribe 转写为文字
  - 输出纯文本文件

流程：
  ┌─ 解析 YouTube URL
  ├─ yt-dlp 多线程下载音频 (原生格式，无需 ffmpeg)
  ├─ 检查文件大小/时长，超阈值则用 PyAV 分片 (无需系统 ffmpeg)
  ├─ 分片并发调用 OpenAI 转写 API (tqdm 进度条)
  └─ 合并结果，保存至 output/{标题}.txt

输入：
  配置文件路径：.env (OPENAI_API_KEY)
  命令行参数：YouTube URL

输出：
  数据文件路径：output/{视频标题}.txt
"""

import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
TEMP_DIR = SCRIPT_DIR / "temp"
COOKIES_FILE = SCRIPT_DIR / "cookies.txt"
MAX_FILE_SIZE = 24 * 1024 * 1024  # 24MB，给 25MB 限制留余量
MODEL_MAX_AUDIO_SECONDS = 1500.0
DEFAULT_MAX_CHUNK_SECONDS = 8 * 60.0
DEFAULT_TRANSCRIBE_WORKERS = 8
MAX_TRANSCRIBE_WORKERS = 16
TRANSCRIBE_RETRIES = 2
CHUNK_OVERLAP_SECONDS = 8.0
TEXT_OVERLAP_SCAN_CHARS = 320
TEXT_OVERLAP_MIN_CHARS = 8
FAST_FRAGMENT_WORKERS = 8
SAFE_FRAGMENT_WORKERS = 1
LIVE_SAFE_STATUSES = {"is_live", "post_live"}


# ======= 下载音频 =======


def should_use_safe_download_mode(url: str) -> tuple[bool, str]:
    """预探测视频状态；直播/直播回放走稳健下载模式。"""
    probe_cmd = [
        sys.executable, "-m", "yt_dlp",
        "--dump-single-json",
        "--skip-download",
        "--no-warnings",
        url,
    ]
    if COOKIES_FILE.exists():
        probe_cmd.extend(["--cookies", str(COOKIES_FILE)])

    try:
        ret = subprocess.run(
            probe_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception:
        return False, ""

    if ret.returncode != 0 or not ret.stdout.strip():
        return False, ""

    try:
        info = json.loads(ret.stdout)
    except json.JSONDecodeError:
        return False, ""

    live_status = str(info.get("live_status") or "")
    was_live = bool(info.get("was_live"))
    if live_status in LIVE_SAFE_STATUSES or was_live:
        return True, live_status
    return False, live_status


def download_audio(url: str) -> tuple[Path, str]:
    """下载 YouTube 音频 (多线程加速)，返回 (文件路径, 视频标题)"""
    TEMP_DIR.mkdir(exist_ok=True)
    use_safe_mode, live_status = should_use_safe_download_mode(url)
    fragment_workers = SAFE_FRAGMENT_WORKERS if use_safe_mode else FAST_FRAGMENT_WORKERS
    if use_safe_mode:
        status_text = live_status or "was_live"
        print(
            f"  检测到直播流 ({status_text})，"
            f"切换稳健下载模式: fragments={fragment_workers}",
            flush=True,
        )

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--remote-components", "ejs:github",
        "-f", "bestaudio[ext=m4a]/bestaudio/best",
        "-o", str(TEMP_DIR / "%(id)s.%(ext)s"),
        "--write-info-json",
        "--concurrent-fragments", str(fragment_workers),
        "--newline",
    ]
    if use_safe_mode:
        cmd.extend([
            "--fragment-retries", "20",
            "--retry-sleep", "fragment:1",
        ])
    # aria2c 多连接加速（如果装了的话）
    if not use_safe_mode and shutil.which("aria2c"):
        cmd.extend([
            "--external-downloader", "aria2c",
            "--external-downloader-args", "-x 16 -s 16 -k 1M",
        ])
    if COOKIES_FILE.exists():
        cmd.extend(["--cookies", str(COOKIES_FILE)])
    cmd.append(url)

    # 不捕获任何输出，进度条/警告/错误全部实时显示
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise RuntimeError("下载失败，请检查上方错误信息")

    # 从 info.json 读取元信息
    json_files = sorted(TEMP_DIR.glob("*.info.json"), key=lambda f: f.stat().st_mtime)
    if not json_files:
        raise RuntimeError("未找到视频信息文件")
    info_path = json_files[-1]
    info = json.loads(info_path.read_text())
    title = info["title"]
    video_id = info["id"]
    ext = info.get("ext", "m4a")
    info_path.unlink()
    return TEMP_DIR / f"{video_id}.{ext}", title


# ======= 音频分片 =======

def resolve_max_chunk_seconds() -> float:
    """读取单片时长上限，默认 8 分钟，且不超过模型 1500 秒限制。"""
    env_val = os.getenv("TRANSCRIBE_MAX_CHUNK_SECONDS")
    if not env_val:
        return DEFAULT_MAX_CHUNK_SECONDS
    try:
        max_seconds = float(env_val)
    except ValueError:
        print("  TRANSCRIBE_MAX_CHUNK_SECONDS 不是数字，回退到默认 480 秒")
        return DEFAULT_MAX_CHUNK_SECONDS
    if max_seconds < 60:
        print("  TRANSCRIBE_MAX_CHUNK_SECONDS 过小，至少使用 60 秒")
        max_seconds = 60.0
    if max_seconds > MODEL_MAX_AUDIO_SECONDS:
        print("  TRANSCRIBE_MAX_CHUNK_SECONDS 超过模型上限，自动下调到 1500 秒")
        max_seconds = MODEL_MAX_AUDIO_SECONDS
    return max_seconds


def split_audio(path: Path) -> list[Path]:
    """按大小/时长分片，确保单片在模型限制内。"""

    import av

    total_size = path.stat().st_size
    max_chunk_seconds = resolve_max_chunk_seconds()
    num_chunks = 1

    inp = av.open(str(path))
    stream = inp.streams.audio[0]
    if inp.duration:
        duration_s = float(inp.duration / 1_000_000)
    else:
        duration_s = float(stream.duration * stream.time_base)
    inp.close()
    if duration_s <= 0:
        raise RuntimeError("无法读取音频时长，无法分片")

    while True:
        chunk_dur_s = duration_s / num_chunks
        overlap_s = min(CHUNK_OVERLAP_SECONDS, max(1.0, chunk_dur_s / 12))
        chunk_window_s = chunk_dur_s + 2 * overlap_s
        estimated_chunk_size = total_size * (chunk_window_s / duration_s)
        size_ok = estimated_chunk_size <= MAX_FILE_SIZE
        duration_ok = chunk_window_s <= max_chunk_seconds
        if size_ok and duration_ok:
            break
        num_chunks += 1

    if num_chunks == 1:
        return [path]

    print(
        f"  自动分为 {num_chunks} 片 "
        f"(每片约 {chunk_dur_s:.0f} 秒，重叠 {overlap_s:.1f} 秒，"
        f"目标上限 {max_chunk_seconds:.0f} 秒)"
    )

    chunks: list[Path] = []
    for idx in range(num_chunks):
        chunk_path = path.parent / f"{path.stem}_c{idx}{path.suffix}"
        start_s = max(0.0, idx * chunk_dur_s - overlap_s)
        end_s = min(duration_s, (idx + 1) * chunk_dur_s + overlap_s)

        inp_chunk = av.open(str(path))
        in_stream = inp_chunk.streams.audio[0]
        out = av.open(str(chunk_path), "w")
        # PyAV 16 使用 add_stream_from_template，旧版本仍支持 add_stream(template=...)
        if hasattr(out, "add_stream_from_template"):
            out_stream = out.add_stream_from_template(in_stream)
        else:
            out_stream = out.add_stream(template=in_stream)

        base_dts = None
        wrote_packet = False
        for packet in inp_chunk.demux(in_stream):
            if packet.dts is None:
                continue
            ts = float(packet.dts * in_stream.time_base)
            if ts < start_s:
                continue
            if ts > end_s:
                break
            if base_dts is None:
                base_dts = packet.dts
            packet.dts -= base_dts
            if packet.pts is not None:
                packet.pts -= base_dts
            packet.stream = out_stream
            out.mux(packet)
            wrote_packet = True

        out.close()
        inp_chunk.close()
        if wrote_packet and chunk_path.exists() and chunk_path.stat().st_size > 0:
            chunks.append(chunk_path)
        else:
            chunk_path.unlink(missing_ok=True)

    if not chunks:
        raise RuntimeError("音频分片失败：未写出有效分片")
    return chunks


# ======= 转写 =======

def resolve_transcribe_workers(chunk_count: int) -> int:
    """计算并发数，支持环境变量 TRANSCRIBE_CONCURRENCY 覆盖。"""
    env_val = os.getenv("TRANSCRIBE_CONCURRENCY")
    if env_val:
        try:
            workers = int(env_val)
        except ValueError:
            print("  TRANSCRIBE_CONCURRENCY 不是整数，回退到默认并发")
        else:
            workers = max(1, workers)
            workers = min(workers, MAX_TRANSCRIBE_WORKERS)
            return min(workers, chunk_count)
    return min(chunk_count, DEFAULT_TRANSCRIBE_WORKERS)


def transcribe_one(path: Path) -> str:
    """单片转写，遇到临时错误会重试。"""
    client = OpenAI()
    for attempt in range(TRANSCRIBE_RETRIES + 1):
        try:
            with open(path, "rb") as f:
                result = client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=f,
                )
            return result.text
        except Exception as exc:  # noqa: BLE001
            if attempt >= TRANSCRIBE_RETRIES:
                raise RuntimeError(f"分片转写失败: {path.name}") from exc
            sleep_s = 1.5 * (attempt + 1)
            print(f"  分片 {path.name} 暂时失败，{sleep_s:.1f}s 后重试...")
            time.sleep(sleep_s)
    raise RuntimeError(f"分片转写失败: {path.name}")


def _find_text_overlap(left: str, right: str) -> int:
    """返回 left 尾部与 right 头部可拼接的重叠长度。"""
    left = left[-TEXT_OVERLAP_SCAN_CHARS:]
    right = right[:TEXT_OVERLAP_SCAN_CHARS]
    max_len = min(len(left), len(right))
    for size in range(max_len, TEXT_OVERLAP_MIN_CHARS - 1, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def merge_chunk_texts(texts: list[str]) -> str:
    """合并分片文本：优先去掉重叠重复，找不到重叠时换行拼接。"""
    merged = ""
    for text in texts:
        current = text.strip()
        if not current:
            continue
        if not merged:
            merged = current
            continue
        overlap = _find_text_overlap(merged, current)
        if overlap:
            merged += current[overlap:]
        else:
            merged += "\n" + current
    return merged


def transcribe(paths: list[Path]) -> str:
    """调用 OpenAI API 转写音频（多片并发，结果按原顺序合并）。"""
    if not paths:
        return ""

    workers = resolve_transcribe_workers(len(paths))
    if workers > 1:
        print(f"  启用并发转写: {workers} 路")

    texts = [""] * len(paths)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(transcribe_one, path): idx
            for idx, path in enumerate(paths)
        }
        with tqdm(total=len(paths), desc="转写进度", unit="片", ncols=70,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            for fut in as_completed(futures):
                idx = futures[fut]
                texts[idx] = fut.result()
                pbar.update(1)
    return merge_chunk_texts(texts)


# ======= 输出文件名 =======

def build_output_filename(title: str, suffix: str = ".txt") -> str:
    """生成可落盘的输出文件名，避免超长标题触发文件名长度限制。"""
    probe_dir = OUTPUT_DIR if OUTPUT_DIR.exists() else SCRIPT_DIR
    try:
        name_max = os.pathconf(probe_dir, "PC_NAME_MAX")
    except (OSError, ValueError, AttributeError):
        name_max = 255

    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
    safe = " ".join(safe.split()).strip(" ._-")
    if not safe:
        safe = "transcript"

    budget = max(1, name_max - len(suffix.encode("utf-8")))
    raw = safe.encode("utf-8")
    if len(raw) > budget:
        safe = raw[:budget].decode("utf-8", errors="ignore").rstrip(" ._-")
        if not safe:
            safe = "transcript"
        while len(safe.encode("utf-8")) > budget:
            safe = safe[:-1]
    return f"{safe}{suffix}"


# ======= 清理临时文件 =======

def cleanup(chunks: list[Path], audio: Path):
    for p in chunks:
        p.unlink(missing_ok=True)
    audio.unlink(missing_ok=True)
    if TEMP_DIR.exists() and not list(TEMP_DIR.iterdir()):
        TEMP_DIR.rmdir()


# ======= 主流程 =======

def main():
    if len(sys.argv) < 2:
        print("用法: python transcribe.py <YouTube URL>")
        sys.exit(1)

    url = sys.argv[1]

    print(f"[1/3] 下载音频: {url}")
    audio_path, title = download_audio(url)
    size_mb = audio_path.stat().st_size / 1024 / 1024
    print(f"  标题: {title}")
    print(f"  大小: {size_mb:.1f} MB")

    print(f"[2/3] 分片检查...")
    chunks = split_audio(audio_path)
    print(f"  共 {len(chunks)} 片")

    print(f"[3/3] 转写中...")
    text = transcribe(chunks)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / build_output_filename(title)
    out_path.write_text(text, encoding="utf-8")
    print(f"完成: {out_path}")

    cleanup(chunks, audio_path)


if __name__ == "__main__":
    main()
