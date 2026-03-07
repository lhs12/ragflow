# ASR 配置示例

## 概述

视频高级处理方案支持两种 ASR 服务：
1. **Whisper** - 推荐，支持多语言、说话人分离、词级时间戳
2. **图灵 ASR** - 需要部署专用服务

## 配置参数

在 `parser_config` 中添加以下参数：

```json
{
  "use_advanced_video_processing": true,
  "enable_asr": true,
  "asr_type": "whisper",
  "asr_api_url": "http://0.0.0.0:9000/asr",
  "asr_language": "zh",
  "speaker_separation": true,
  "frame_interval": 2.0,
  "enable_ocr": true,
  "enable_vlm": false
}
```

## 参数说明

### 基础参数
- `use_advanced_video_processing`: 是否启用高级视频处理（默认 false）
- `enable_asr`: 是否启用语音识别（默认 true）
- `enable_ocr`: 是否启用 OCR（默认 true）
- `enable_vlm`: 是否启用视觉语言模型（默认 false）

### ASR 参数
- `asr_type`: ASR 类型，可选 `"whisper"` 或 `"tuling"`（默认 `"whisper"`）
- `asr_api_url`: ASR 服务地址（可选，不指定则使用默认值）
- `asr_language`: 语言代码（Whisper 专用），如 `"zh"`, `"en"`, `"auto"`（默认 `"auto"`）
- `speaker_separation`: 是否启用说话人分离（默认 false）

### 抽帧参数
- `frame_interval`: 抽帧间隔（秒），默认 1.0
- `use_keyframe`: 是否使用关键帧提取（默认 false）

## Whisper 配置示例

### 基础转写（中文）
```json
{
  "use_advanced_video_processing": true,
  "asr_type": "whisper",
  "asr_language": "zh",
  "frame_interval": 2.0
}
```

### 说话人分离（英文）
```json
{
  "use_advanced_video_processing": true,
  "asr_type": "whisper",
  "asr_language": "en",
  "speaker_separation": true,
  "min_speakers": 2,
  "max_speakers": 4
}
```

### 自定义 Whisper 服务地址
```json
{
  "use_advanced_video_processing": true,
  "asr_type": "whisper",
  "asr_api_url": "http://your-whisper-server:9000/asr",
  "asr_language": "auto"
}
```

## 图灵 ASR 配置示例

```json
{
  "use_advanced_video_processing": true,
  "asr_type": "tuling",
  "asr_api_url": "http://112.39.27.207:8900/tuling/asr/v3/process",
  "speaker_separation": true
}
```

**注意**：图灵 ASR 目前返回模拟数据，需要部署服务后才能正常使用。

## Whisper 支持的语言

常用语言代码：
- `zh` - 中文
- `en` - 英文
- `es` - 西班牙语
- `fr` - 法语
- `de` - 德语
- `ja` - 日语
- `ko` - 韩语
- `auto` - 自动检测

完整语言列表请参考 [Whisper 文档](https://github.com/openai/whisper#available-models-and-languages)。

## Whisper 高级参数

### 词级时间戳（Faster Whisper）
```json
{
  "asr_type": "whisper",
  "word_timestamps": true
}
```

### VAD 过滤（Faster Whisper）
```json
{
  "asr_type": "whisper",
  "vad_filter": true
}
```

### 翻译任务（转为英文）
```json
{
  "asr_type": "whisper",
  "task": "translate"
}
```

## 输出格式

ASR 返回的每个片段包含：
- `begin_sec`: 开始时间（秒）
- `end_sec`: 结束时间（秒）
- `text`: 转写文本
- `spk`: 说话人序号（启用说话人分离时）
- `confidence`: 置信度（可选）

## 故障排查

### Whisper 服务连接失败
检查服务是否启动：
```bash
curl http://0.0.0.0:9000/asr
```

### 音频提取失败
确保容器内安装了 ffmpeg：
```bash
docker exec -it ragflow-server bash
ffmpeg -version
```

### 转写结果为空
- 检查视频是否包含音频轨道
- 尝试调整 `asr_language` 参数
- 查看日志中的错误信息
