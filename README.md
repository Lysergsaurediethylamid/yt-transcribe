# yt-transcribe

YouTube 视频音频转写工具。下载音频，调用 OpenAI `gpt-4o-mini-transcribe` 转写为纯文本。

## 流程

```
┌─ 解析 YouTube URL
├─ yt-dlp 多线程下载音频 (原生格式，无需 ffmpeg)
├─ 检查文件大小/时长，超阈值则用 PyAV 分片
├─ 分片并发调用 OpenAI 转写 API
└─ 合并结果，保存至 output/{标题}.txt
```

## 安装

```bash
pip install -r requirements.txt
```

依赖：`yt-dlp` `openai` `av` `python-dotenv` `tqdm`

可选：安装 `aria2c` 可进一步加速下载。

## 配置

复制 `.env.example` 为 `.env`，填入 OpenAI API Key：

```bash
cp .env.example .env
```

```
OPENAI_API_KEY=sk-proj-your-key-here
```

可选环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TRANSCRIBE_CONCURRENCY` | `8` | 转写并发数 (上限 16) |
| `TRANSCRIBE_MAX_CHUNK_SECONDS` | `480` | 单片时长上限，秒 (上限 1500) |

## 使用

```bash
python transcribe.py <YouTube URL>
```

输出保存至 `output/{视频标题}.txt`。

## 特性

- 自动检测直播/直播回放，切换稳健下载模式
- 大文件自动分片，分片间带重叠以避免文字截断
- 合并时自动去除重叠重复文本
- 支持 `cookies.txt` 访问需登录的视频（Netscape 格式，放在项目根目录）
