# RAGFlow 视频处理方案设计文档

## 一、方案概述

本方案实现视频文件的多模态处理，包括：
1. 视频抽帧
2. 画面文字提取（OCR）
3. 音频转写（ASR）
4. 多模态内容融合
5. 检索与召回

## 二、整体架构

```
视频文件
  ↓
1. 视频抽帧（OpenCV/ffmpeg）
  ↓
2. 音视频分离（ffmpeg）
  ↓
3. 并行处理：
   ├─ 音频转写（图灵ASR服务）→ 带时间戳的转写文本
   └─ 每帧处理：
      ├─ OCR提取画面文字
      └─ VLM描述（可选）
  ↓
4. 时间戳对齐 + 多模态内容融合
  ↓
5. 生成chunk列表（每帧一个chunk）
  ↓
6. 存储到ES + MinIO
```

## 三、数据结构设计

### 3.1 Chunk数据结构

每一帧生成一个chunk，完全复用现有的图片处理机制：

```python
{
    # 基础字段（复用现有）
    "docnm_kwd": "video.mp4",           # 视频文件名
    "doc_id": "video_doc_id",           # 视频文档ID
    "doc_type_kwd": "video_frame",      # 新增类型
    "title_tks": ["video", "mp4"],      # 文件名分词

    # 图片字段（复用现有机制）
    "image": <PIL.Image>,               # 帧图片，后续通过image2id()转为img_id
    "img_id": "kb_id-chunk_id",         # 由task_executor自动生成

    # 视频帧特有字段
    "frame_index_int": 5,               # 帧序号（从0开始）
    "frame_timestamp_flt": 10.0,        # 帧时间戳（秒）
    "frame_duration_flt": 2.0,          # 该帧代表的时间段长度（秒）

    # 多模态融合内容
    "content_with_weight": """
[时间: 00:00:10-00:00:12 | 第5帧]

[画面文字(OCR)]
产品介绍
价格：99元

[语音内容(ASR)]
[说话人0] 大家好，今天给大家介绍这款产品
[说话人1] 它的价格非常实惠，只要99元

[画面描述(VLM)]
画面显示一个产品展示页面，中央是产品图片，下方有价格标签。
    """,

    # 分词字段（自动生成）
    "content_ltks": "...",              # 内容分词
    "content_sm_ltks": "...",           # 细粒度分词

    # 元数据（用于前端展示）
    "frame_timestamp_flt": 10.0,        # 帧时间戳（秒）
    "frame_duration_flt": 2.0,          # 该帧代表的时间段长度（秒）
}
```

### 3.2 字段说明

| 字段 | 类型 | 说明 | 是否新增 |
|------|------|------|---------|
| `doc_type_kwd` | string | 值为 `"video_frame"` | ✅ 新增值 |
| `image` | PIL.Image | 帧图片对象 | ❌ 复用 |
| `img_id` | string | 图片ID，格式 `bucket-objname` | ❌ 复用 |
| `frame_index_int` | int | 帧序号（从0开始） | ✅ 新增 |
| `frame_timestamp_flt` | float | 帧时间戳（秒） | ✅ 新增 |
| `frame_duration_flt` | float | 时间段长度（秒） | ✅ 新增 |

## 四、核心实现

### 4.1 文件结构

```
rag/
├── app/
│   └── picture.py              # 修改：增加视频处理逻辑
├── utils/
│   └── asr_client.py           # 新增：图灵ASR客户端
└── flow/
    └── parser/
        └── parser.py           # 修改：增加视频配置
```

### 4.2 ASR客户端实现

**文件**: `rag/utils/asr_client.py`

```python
class TulingASRClient:
    """图灵ASR转写客户端"""

    def __init__(self, api_url: str, timeout: int = 300):
        self.api_url = api_url
        self.timeout = timeout

    def transcribe(self, audio_path: str, audio_id: str = None,
                   speaker_separation: bool = False) -> List[Dict]:
        """
        调用ASR服务进行转写

        Returns:
            [
                {
                    "lid": 片段ID,
                    "spk": 说话人序号,
                    "begin_sec": 开始时间（秒）,
                    "end_sec": 结束时间（秒）,
                    "text": 转写文本,
                    "confidence": 置信度
                }
            ]
        """
```

**关键功能**：
1. 音频格式转换（转为WAV）
2. 音频文件上传（获取HTTP可访问URI）
3. 调用图灵ASR API
4. 时间单位转换（帧 → 秒）
5. 解析转写结果

### 4.3 视频处理流程

**文件**: `rag/app/picture.py`

```python
def chunk(filename, binary, tenant_id, lang, callback=None, **kwargs):
    """视频处理主函数"""

    # 1. 抽帧
    frames, duration = extract_frames_from_video(binary, frame_interval, max_frames)

    # 2. 音视频分离 + 转写
    extract_audio_from_video(binary, audio_path)
    transcript_segments = transcribe_audio_file(audio_path, asr_api_url)

    # 3. 处理每一帧
    for frame_data in frames:
        # 3.1 OCR提取文字
        ocr_text = ocr(frame_data['image'])

        # 3.2 VLM描述（可选）
        vlm_desc = vlm_model.describe(frame_data['image'])

        # 3.3 匹配音频转写
        asr_text = match_transcript_to_frame(
            frame_data['timestamp'],
            frame_data['duration'],
            transcript_segments
        )

        # 3.4 融合多模态内容
        content = format_multimodal_content(ocr_text, asr_text, vlm_desc)

        # 3.5 构建chunk
        chunk = {
            "doc_type_kwd": "video_frame",
            "image": frame_data['image'],
            "frame_index_int": frame_data['frame_index'],
            "frame_timestamp_flt": frame_data['timestamp'],
            "frame_duration_flt": frame_data['duration'],
            "content_with_weight": content
        }
        tokenize(chunk, content, eng)
        chunks.append(chunk)

    return chunks
```

### 4.4 时间戳对齐算法

```python
def match_transcript_to_frame(frame_timestamp, frame_duration, transcript_segments):
    """
    将ASR转写片段匹配到视频帧

    算法：判断时间段是否有重叠
    - 帧时间段: [frame_start, frame_end]
    - 转写片段: [seg_start, seg_end]
    - 重叠条件: NOT (seg_end < frame_start OR seg_start > frame_end)
    """
    frame_start = frame_timestamp
    frame_end = frame_timestamp + frame_duration

    matched_texts = []
    for seg in transcript_segments:
        if not (seg["end_sec"] < frame_start or seg["begin_sec"] > frame_end):
            text = seg["text"]
            if seg.get("spk") is not None:
                text = f"[说话人{seg['spk']}] {text}"
            matched_texts.append(text)

    return " ".join(matched_texts)
```

## 五、配置参数

### 5.1 Parser配置

**文件**: `rag/flow/parser/parser.py`

```python
"video": {
    "parse_method": "multimodal_extraction",
    "llm_id": "",                       # VLM模型ID（可选）
    "frame_interval": 2.0,              # 抽帧间隔（秒）
    "max_frames": 100,                  # 最大帧数
    "enable_ocr": True,                 # 启用OCR
    "enable_asr": True,                 # 启用音频转写
    "enable_vlm": False,                # 启用VLM描述（可选，成本高）
    "speaker_separation": False,        # 是否进行说话人分离
    "asr_api_url": "http://112.39.27.207:8900/tuling/asr/v3/process",
    "output_format": "json",
}
```

### 5.2 前端配置界面

**文件**: `web/src/pages/agent/form/parser-form/video-form-fields.tsx`

```typescript
export function VideoFormFields({ prefix }: OutputFormatFormFieldProps) {
  return (
    <>
      {/* 抽帧配置 */}
      <FormField label="抽帧间隔(秒)">
        <InputNumber min={0.5} max={10} step={0.5} defaultValue={2} />
      </FormField>

      <FormField label="最大帧数">
        <InputNumber min={10} max={500} defaultValue={100} />
      </FormField>

      {/* 多模态处理开关 */}
      <FormField label="启用OCR">
        <Switch defaultChecked />
      </FormField>

      <FormField label="启用音频转写">
        <Switch defaultChecked />
      </FormField>

      <FormField label="启用VLM描述">
        <Switch />
      </FormField>

      <FormField label="说话人分离">
        <Switch />
      </FormField>

      {/* ASR服务配置 */}
      <FormField label="ASR服务地址">
        <Input placeholder="http://..." />
      </FormField>
    </>
  );
}
```

## 六、检索与召回

### 6.1 检索机制

**完全复用现有检索系统**，无需任何改动：

1. ES自动索引 `doc_type_kwd: "video_frame"` 的chunk
2. 通过 `content_with_weight` 字段进行全文检索
3. 返回结果包含 `img_id`、`frame_index_int`、`frame_timestamp_flt` 等字段

### 6.2 召回关联

**通过现有字段实现视频关联**：

| 需求 | 实现方式 |
|------|---------|
| 知道帧属于哪个视频 | 通过 `doc_id` 字段 |
| 获取视频文件名 | 通过 `docnm_kwd` 字段 |
| 获取帧序号 | 通过 `frame_index_int` 字段 |
| 获取时间戳 | 通过 `frame_timestamp_flt` |
| 获取帧图片 | 通过 `img_id` 调用 `/api/document/image/<image_id>` |

### 6.3 前端展示

#### 6.3.1 Chunk卡片展示

**文件**: `web/src/pages/chunk/chunk-card.tsx`

```typescript
export function ChunkCard({ chunk }: { chunk: IChunk }) {
  if (chunk.doc_type_kwd === 'video_frame') {
    const frameIndex = chunk.frame_index_int ?? 0;
    const timestamp = chunk.frame_timestamp_flt ?? 0;

    return (
      <Card>
        <CardContent className="p-4">
          <div className="flex gap-4">
            {/* 帧图片 - 复用现有接口 */}
            <div className="relative">
              <img
                src={`/api/document/image/${chunk.image_id}`}
                className="w-48 h-32 object-cover rounded"
              />
              <div className="absolute bottom-2 right-2 bg-black/70 text-white px-2 py-1 rounded text-xs">
                {formatTime(timestamp)}
              </div>
            </div>

            {/* 内容 */}
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <Badge>第{frameIndex}帧</Badge>
                <span className="text-sm text-gray-500">
                  {chunk.doc_name}
                </span>
                {/* 跳转到视频 */}
                <Button
                  size="sm"
                  onClick={() => jumpToVideo(chunk.doc_id, timestamp)}
                >
                  <PlayCircle className="w-4 h-4 mr-1" />
                  跳转到视频
                </Button>
              </div>

              {/* 多模态内容 */}
              <div className="text-sm whitespace-pre-wrap">
                {chunk.content_with_weight}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  // 原有逻辑...
}
```

#### 6.3.2 视频详情页

**文件**: `web/src/pages/document/video-detail.tsx`

```typescript
export function VideoDetail({ documentId }: { documentId: string }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [chunks, setChunks] = useState<IChunk[]>([]);
  const [currentTime, setCurrentTime] = useState(0);

  // 从URL获取时间戳参数
  const searchParams = new URLSearchParams(window.location.search);
  const initialTime = parseFloat(searchParams.get('t') || '0');

  useEffect(() => {
    // 加载视频的所有帧chunk
    loadVideoChunks(documentId).then(setChunks);

    // 跳转到指定时间
    if (videoRef.current && initialTime > 0) {
      videoRef.current.currentTime = initialTime;
    }
  }, [documentId, initialTime]);

  return (
    <div className="grid grid-cols-2 gap-4">
      {/* 左侧：视频播放器 */}
      <div>
        <video
          ref={videoRef}
          src={`/api/document/video/${documentId}`}
          controls
          onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
        />
      </div>

      {/* 右侧：帧列表（时间轴） */}
      <div className="space-y-2 overflow-y-auto">
        {chunks.map((chunk, index) => (
          <Card
            key={chunk.chunk_id}
            className={isCurrentFrame(chunk, currentTime) ? 'border-blue-500' : ''}
          >
            <CardContent className="p-2">
              <div className="flex gap-2">
                <img
                  src={`/api/document/image/${chunk.image_id}`}
                  className="w-24 h-16 object-cover cursor-pointer"
                  onClick={() => {
                    videoRef.current.currentTime = chunk.frame_timestamp_flt;
                  }}
                />
                <div className="flex-1 text-xs">
                  <div className="font-semibold">
                    {formatTime(chunk.frame_timestamp_flt)}
                  </div>
                  <div className="text-gray-600 line-clamp-3">
                    {chunk.content_with_weight}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
```

## 七、技术依赖

### 7.1 Python依赖

```python
# 已有依赖
opencv-python      # 视频抽帧
Pillow            # 图片处理
numpy             # 数组处理

# 需要确认
ffmpeg-python     # 音视频处理（或直接调用ffmpeg命令）
requests          # HTTP请求
```

### 7.2 系统依赖

```bash
# FFmpeg（音视频处理）
apt-get install ffmpeg  # Ubuntu/Debian
brew install ffmpeg     # macOS
```

### 7.3 外部服务

- **图灵ASR服务**: `http://112.39.27.207:8900/tuling/asr/v3/process`
- **MinIO**: 存储帧图片和音频文件

## 八、实施步骤

### Phase 1: 基础抽帧（1-2天）
- [ ] 实现视频抽帧逻辑
- [ ] 实现音视频分离
- [ ] 测试基本流程

### Phase 2: ASR集成（2-3天）
- [ ] 实现ASR客户端
- [ ] 实现音频文件上传
- [ ] 实现时间戳对齐
- [ ] 测试ASR转写

### Phase 3: 多模态融合（1-2天）
- [ ] 集成OCR处理
- [ ] 集成VLM描述（可选）
- [ ] 实现内容融合逻辑
- [ ] 测试完整流程

### Phase 4: 前端展示（2-3天）
- [ ] 实现Chunk卡片展示
- [ ] 实现视频详情页
- [ ] 实现时间轴交互
- [ ] 测试用户体验

### Phase 5: 优化与测试（2-3天）
- [ ] 性能优化
- [ ] 错误处理完善
- [ ] 端到端测试
- [ ] 文档完善

## 九、关键技术点

### 9.1 音频文件上传

**问题**: ASR服务需要HTTP可访问的音频URI

**解决方案**:
```python
def upload_audio_to_storage(self, audio_path: str) -> str:
    """上传音频到MinIO并返回公开访问URL"""
    from common import settings

    bucket = "audio-temp"
    filename = f"{uuid.uuid4()}.wav"

    with open(audio_path, 'rb') as f:
        settings.STORAGE_IMPL.put(bucket, filename, f.read())

    # 返回MinIO公开访问URL
    return f"http://{settings.MINIO_HOST}:{settings.MINIO_PORT}/{bucket}/{filename}"
```

### 9.2 时间单位转换

**问题**: ASR返回的是帧数，需要转换为秒

**解决方案**:
```python
# 图灵ASR的帧率（需要确认实际值）
FRAME_RATE = 100.0  # 100帧/秒

begin_sec = lattice["begin"] / FRAME_RATE
end_sec = lattice["end"] / FRAME_RATE
```

### 9.3 说话人分离

**配置**:
```python
request_data = {
    "audio": {
        "spkn": 2 if speaker_separation else 1,  # 1不分离 2分离
        # ...
    }
}
```

**展示**:
```python
if seg.get("spk") is not None:
    text = f"[说话人{seg['spk']}] {text}"
```

## 十、优化建议

### 10.1 性能优化

1. **并行处理**: OCR和VLM可以批量并发
2. **异步处理**: ASR转写可以异步进行
3. **缓存机制**: 相同视频避免重复处理
4. **分段处理**: 大视频分段处理，避免内存溢出

### 10.2 成本控制

1. **VLM可选**: 默认关闭，按需启用
2. **抽帧策略**: 智能关键帧检测，减少冗余帧
3. **OCR优先**: 优先使用OCR，VLM作为补充

### 10.3 准确性提升

1. **场景变化检测**: 只在场景变化时抽帧
2. **时间戳精确对齐**: 使用更精确的时间戳匹配算法
3. **多模态权重**: 根据置信度调整不同模态的权重

## 十一、注意事项

### 11.1 必须实现的功能

1. ✅ **音频文件上传**: 必须实现 `upload_audio_to_storage()` 函数
2. ✅ **时间单位转换**: 确认ASR服务的帧率，正确转换时间
3. ✅ **错误处理**: ASR失败时不中断整个流程

### 11.2 需要配置的参数

1. ASR服务地址
2. MinIO访问配置
3. 帧率转换参数
4. 抽帧间隔和最大帧数

### 11.3 需要测试的场景

1. 短视频（<1分钟）
2. 长视频（>10分钟）
3. 无音频视频
4. 多说话人视频
5. 画面文字较多的视频

## 十二、总结

本方案的核心优势：

1. ✅ **完全复用现有架构**: 无需修改ES schema、检索系统、图片存储
2. ✅ **多模态融合**: OCR + ASR + VLM，提供丰富的检索内容
3. ✅ **灵活配置**: 用户可按需启用不同的处理方式
4. ✅ **自然关联**: 通过现有字段实现视频帧与原视频的关联
5. ✅ **渐进式实现**: 可分阶段实施，逐步完善功能

关键创新点：

1. 使用独立的 `frame_index_int` 字段存储帧序号
2. 使用 `frame_timestamp_flt` 存储精确时间戳
3. 多模态内容融合到 `content_with_weight`
4. 完全复用 `image2id()` 和图片接口

## 十三、现有视频处理大模型兼容性问题

### 13.1 现状分析

RAGFlow现有的视频处理逻辑（`rag/app/picture.py`）通过 `cv_mdl.async_chat(video_bytes=...)` 调用大模型对整个视频生成摘要。但在 `rag/llm/cv_model.py` 中，**只有2个模型类实现了视频处理**，其余模型类（包括GPUStackCV）都不支持。

### 13.2 各模型类视频处理能力

| 模型类 | 继承自 | 是否支持视频 | 实现方式 |
|--------|--------|-------------|---------|
| **QWenCV** | GptV4 | ✅ 支持 | 重写 `async_chat` + `_process_video`，使用 DashScope MultiModalConversation API，将视频保存为临时文件通过 `file://` 传递 |
| **GeminiCV** | Base | ✅ 支持 | 重写 `async_chat` + `_process_video`，≤20MB用inline_data，>20MB用Files API上传，使用 `gemini-2.5-flash` |
| **GPUStackCV** | GptV4 | ❌ 不支持 | 仅重写 `__init__`，继承GptV4的 `async_chat`，不处理 `video_bytes` 参数 |
| **GptV4** (OpenAI) | Base | ❌ 不支持 | `async_chat` 只处理 `images` 参数，忽略 `video_bytes` |
| **AzureGptV4** | GptV4 | ❌ 不支持 | 同GptV4 |
| **xAICV** | GptV4 | ❌ 不支持 | 同GptV4 |
| **HunyuanCV** | GptV4 | ❌ 不支持 | 同GptV4 |
| **StepFunCV** | GptV4 | ❌ 不支持 | 同GptV4 |
| **VolcEngineCV** | GptV4 | ❌ 不支持 | 同GptV4 |
| **OllamaCV** | Base | ❌ 不支持 | 独立实现，无视频处理 |
| **NvidiaCV** | Base | ❌ 不支持 | 独立实现，无视频处理 |
| **AnthropicCV** | Base | ❌ 不支持 | 独立实现，无视频处理 |
| **MoonshotCV** | GptV4 | ❌ 不支持 | 同GptV4 |
| **LocalAICV** | GptV4 | ❌ 不支持 | 同GptV4 |
| **XinferenceCV** | GptV4 | ❌ 不支持 | 同GptV4 |
| **LmStudioCV** | GptV4 | ❌ 不支持 | 同GptV4 |
| **OpenAI_APICV** | GptV4 | ❌ 不支持 | 同GptV4 |
| **SILICONFLOWCV** | GptV4 | ❌ 不支持 | 同GptV4 |
| **OpenRouterCV** | GptV4 | ❌ 不支持 | 同GptV4 |

### 13.3 根本原因

```
Base.async_chat(system, history, gen_conf, images=None, **kwargs)
  ↓
  只处理 images 参数，**kwargs 中的 video_bytes 被忽略

GptV4 继承 Base，未重写 async_chat
  ↓
GPUStackCV 继承 GptV4，未重写 async_chat
  ↓
调用 async_chat(video_bytes=binary) 时，video_bytes 被 **kwargs 吞掉，无任何处理
```

### 13.4 本方案的解决方式

本方案通过**抽帧 + OCR + ASR**的方式处理视频，**不再依赖大模型直接处理视频**：

```
旧方案：视频 → 大模型(video_bytes) → 摘要文本
                ↑ 只有QWenCV和GeminiCV支持

新方案：视频 → 抽帧 → 每帧图片 → OCR提取文字
             → 音频 → ASR转写文字
             → (可选) 每帧图片 → VLM描述（describe方法，所有模型都支持）
```

**关键区别**：
- 旧方案调用 `async_chat(video_bytes=binary)`，需要模型原生支持视频输入
- 新方案调用 `describe(image_bytes)` 或 `describe_with_prompt(image_bytes)`，这是所有CV模型类都实现了的方法
- OCR使用RAGFlow内置的OCR引擎，不依赖任何大模型
- ASR使用独立的图灵ASR服务，不依赖CV模型

### 13.5 GPUStackCV的适配

在新方案下，GPUStackCV **无需任何修改** 即可参与视频处理：

| 处理环节 | 调用方法 | GPUStackCV是否支持 |
|---------|---------|-------------------|
| OCR | `ocr(np.array(image))` | 不涉及（内置引擎） |
| ASR | 图灵ASR HTTP接口 | 不涉及（独立服务） |
| VLM描述（可选） | `cv_mdl.describe(image_bytes)` | ✅ 继承自GptV4，已实现 |
| VLM带提示词描述（可选） | `cv_mdl.describe_with_prompt(image_bytes, prompt)` | ✅ 继承自GptV4，已实现 |

### 13.6 如果仍需保留旧的视频摘要功能

如果某些场景仍需要大模型直接处理整个视频（如生成视频整体摘要），可以在配置中区分：

```python
"video": {
    "parse_method": "multimodal_extraction",  # 新方案：抽帧+OCR+ASR
    # "parse_method": "llm_summary",          # 旧方案：大模型直接处理视频（仅QWenCV/GeminiCV可用）
}
```

当 `parse_method` 为 `"llm_summary"` 时，走原有逻辑；为 `"multimodal_extraction"` 时，走新的抽帧方案。
