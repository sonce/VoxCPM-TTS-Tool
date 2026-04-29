# VoxCPM TTS Tool

一个基于 Gradio 的 VoxCPM 文本转语音工具。项目封装了
[`voxcpm`](https://pypi.org/project/voxcpm/) SDK，提供可视化的语音合成、
音色管理、多角色脚本、长文本切分、参考音频转写、参考音频降噪和 `.wav`
结果下载。

## 功能特性

- Web UI：启动后在浏览器中使用，无需单独写推理脚本。
- 三种音色模式：提示词音色设计、可控克隆、高保真克隆。
- 音色库：保存、编辑、删除可复用音色，参考音频保存在本地。
- 多角色脚本：支持在文本中用 `<音色名>` 切换说话人。
- 长文本处理：自动切分并逐段生成，最终导出音频文件。
- 参考音频转写：使用 SenseVoiceSmall 为高保真克隆填写 transcript。
- 参考音频降噪：可选使用 ZipEnhancer，对克隆工作流的参考音频降噪。
- 本地模型缓存：首次启动自动下载模型，后续复用 `pretrained_models/`。

完整设计说明见
[`docs/superpowers/specs/2026-04-26-voxcpm-tts-tool-design.md`](docs/superpowers/specs/2026-04-26-voxcpm-tts-tool-design.md)。

## 环境要求

- Python 3.10、3.11 或 3.12。
- 不建议使用 Python 3.13：当前依赖链中的 `editdistance` 在 3.13 上可能构建失败。
- 首次下载模型约需 6 GB 磁盘空间。
- 首次启动需要网络访问 ModelScope；VoxCPM2 和 SenseVoiceSmall 也支持 Hugging Face 回退下载。
- 可选安装 `ffmpeg`：用于更稳定、更快地加载参考音频。

## 快速启动

### Windows PowerShell

```powershell
.\run.ps1
```

默认监听所有网卡。浏览器本机访问：

```text
http://127.0.0.1:8808
```

常用参数：

```powershell
.\run.ps1 --port 9000
.\run.ps1 --host 127.0.0.1
.\run.ps1 --share
.\run.ps1 --root E:\path\to\data-root
```

### Linux / macOS / Git Bash

```bash
./run.sh
```

常用参数：

```bash
./run.sh --port 9000
./run.sh --host 127.0.0.1
./run.sh --share
./run.sh --root /path/to/data-root
```

`run.ps1` 和 `run.sh` 会自动完成：

1. 优先使用 `uv` 创建 Python 3.12 虚拟环境并安装依赖。
2. 如果没有 `uv`，回退到系统里的 Python 3.10-3.12 + `venv` + `pip`。
3. 创建或复用 `.venv`。
4. 安装 `requirements.txt` 和本地包。
5. 启动 `app.py`。

首次启动会把模型下载到 `pretrained_models/`，后续启动会复用本地缓存。

## 推荐安装 uv

安装 `uv` 后，启动脚本可以自动准备独立的 Python 3.12 环境。

Windows PowerShell：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Linux / macOS：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 手动安装

不使用启动脚本时，可以手动创建环境：

Windows：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

Linux / macOS：

```bash
python -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip install -e . --no-deps
```

## 手动运行

Windows：

```powershell
.\.venv\Scripts\python.exe app.py --port 8808
```

Linux / macOS：

```bash
./.venv/bin/python app.py --port 8808
```

`app.py` 参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--port` | `8808` | Gradio 服务端口 |
| `--host` | `0.0.0.0` | Gradio 监听地址；默认允许局域网访问 |
| `--share` | `false` | 启用 Gradio 公网分享链接 |
| `--root` | `.` | 数据根目录，影响 `voices/`、`outputs/`、`pretrained_models/` 的位置 |

## 使用说明

### 1. 选择或创建默认音色

进入 Web UI 后，可以直接使用默认临时音色，也可以在音色管理区域创建可复用音色。

### 2. 选择音色模式

| 模式 | 适用场景 | 需要输入 |
| --- | --- | --- |
| Voice Design / 音色设计 | 通过文字描述控制声音风格 | 音色名称、控制提示词 |
| Controllable Cloning / 可控克隆 | 用参考音频克隆音色，并可附加风格控制 | 参考音频、可选控制提示词 |
| Ultimate Cloning / 高保真克隆 | 参考音频 + 对应文本，尽量还原音色 | 参考音频、参考音频文本 transcript |

高保真克隆中的转写按钮会调用 SenseVoiceSmall 填写 transcript。上传音频不会自动转写，避免覆盖已输入文本。

### 3. 输入脚本并生成

单音色文本可以直接输入。多角色文本可以使用 `<音色名>` 切换音色：

```text
<女声>大家好。[laughing] 今天我们介绍 VoxCPM。
<男声>下面换一个声音。
这一行没有指定音色，所以回到默认音色。
```

脚本规则：

- `<音色名>` 从当前位置开始切换当前行使用的音色。
- 新的一行会重置为界面中选择的默认音色。
- `[laughing]` 这类标签会原样传给 VoxCPM。
- 音色名匹配会去除首尾空白，并忽略大小写。

### 4. 下载结果

生成结果会保存为 `.wav` 文件，并显示在界面中供下载。

## 数据目录

| 路径 | 说明 |
| --- | --- |
| `voices/voices.json` | 音色元数据 |
| `voices/audio/<id>.wav` | 音色参考音频 |
| `outputs/YYYYMMDD-HHMMSS-mmm.wav` | 生成的音频文件 |
| `pretrained_models/VoxCPM2/` | VoxCPM2 模型缓存 |
| `pretrained_models/SenseVoiceSmall/` | SenseVoiceSmall 模型缓存 |
| `pretrained_models/ZipEnhancer/` | ZipEnhancer 模型缓存 |

这些运行时目录通常不需要提交到 Git。

## 模型路径覆盖

如果已经手动下载了模型，可以通过环境变量指定本地路径：

| 环境变量 | 说明 |
| --- | --- |
| `VOXCPM_MODEL_DIR` | VoxCPM2 模型目录 |
| `VOXCPM_ASR_MODEL_DIR` | SenseVoiceSmall 模型目录 |
| `ZIPENHANCER_MODEL_PATH` | ZipEnhancer 模型目录 |

指定的目录必须包含对应模型的有效文件。例如 VoxCPM2 目录中应包含
`config.json`、`model.safetensors` 或 `pytorch_model.bin` 等文件。空目录或只包含
`.gitkeep` 的目录会被视为无效，程序会继续尝试自动下载。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Linux / macOS：

```bash
./.venv/bin/python -m pytest -v
```

测试使用 fake model 和 fake ASR，不会进行真实模型推理，也不需要网络。

## 常见问题

### Windows 运行 `bash run.sh` 报 WSL 错误

如果出现类似：

```text
WSL ... execvpe(/bin/bash) failed: No such file or directory
```

说明当前 `bash` 指向了不可用的 WSL。建议直接使用：

```powershell
.\run.ps1
```

或显式调用 Git Bash：

```powershell
& "C:\Program Files\Git\bin\bash.exe" run.sh
```

### 首次启动很慢

首次启动需要下载 VoxCPM2、SenseVoiceSmall 和 ZipEnhancer，模型总量约 6 GB。下载完成后会缓存到
`pretrained_models/`，再次启动会明显变快。

### 转写按钮不可用

SenseVoiceSmall 或 `funasr` 不可用时，转写按钮会禁用。仍然可以手动填写 transcript 后继续生成。

### 降噪开关没有效果

ZipEnhancer 没有成功加载时，界面仍可生成音频，但降噪开关不会生效。启动日志会提示 ZipEnhancer 的可用状态。
