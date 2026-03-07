# 视频帧输出示例

## 文档结构示例

```json
{
  "docnm_kwd": "demo_video.mp4",
  "title_tks": "demo video",
  "title_sm_tks": "demo video",

  "video_timestamp": 2.5,
  "frame_index": 5,
  "is_video_frame": true,
  "video_filename": "demo_video.mp4",
  "video_duration": 120.5,
  "video_fps": 30.0,

  "asr_text": "[说话人0] 大家好，欢迎观看本视频 [说话人0] 今天我们来介绍一下这个产品的主要功能",
  "visual_text": "产品介绍\n主要功能列表\n1. 功能A\n2. 功能B",

  "frame_image_path": "videos/tenant123/demo_video/frame_5_2.50.jpg",
  "frame_image_base64": "/9j/4AAQSkZJRgABAQAA...",

  "content_ltks": "大家 好 欢迎 观看 本 视频 今天 我们 来 介绍 一下 这个 产品 的 主要 功能 产品 介绍 主要 功能 列表 功能 A 功能 B"
}
```

## 字段说明

### 视频元信息
- `video_timestamp`: 当前帧在视频中的时间位置（秒）
- `frame_index`: 帧序号（从 0 开始）
- `video_duration`: 视频总时长
- `video_fps`: 视频帧率

### 内容字段（分开存储）

#### `asr_text` - 语音转写文本
- 包含该帧时间段内的所有语音内容
- 如果启用说话人分离，会标注 `[说话人N]`
- 多个语音片段用空格连接
- 示例：`"[说话人0] 大家好 [说话人1] 你好"`

#### `visual_text` - 视觉内容文本
- 包含 OCR 识别的文字（如 PPT 文字、字幕等）
- 包含 VLM 描述的画面内容（如果启用）
- OCR 和 VLM 结果用换行符分隔
- 示例：`"产品介绍\n主要功能列表"`

#### `content_ltks` - 检索分词
- 融合 `asr_text` + `visual_text` 后的分词结果
- 用于全文检索和语义匹配

### 图片字段
- `frame_image_path`: MinIO 存储路径
- `frame_image_base64`: base64 编码（可直接用于前端展示）

## 前端展示示例

### React 组件示例

```jsx
function VideoFrameCard({ frame }) {
  return (
    <div className="frame-card">
      {/* 帧图片 */}
      <img
        src={`data:image/jpeg;base64,${frame.frame_image_base64}`}
        alt={`Frame at ${frame.video_timestamp}s`}
      />

      {/* 时间戳 */}
      <div className="timestamp">
        {formatTime(frame.video_timestamp)}
      </div>

      {/* 语音内容区域 */}
      {frame.asr_text && (
        <div className="audio-section">
          <h4>🎤 语音内容</h4>
          <p>{frame.asr_text}</p>
        </div>
      )}

      {/* 视觉内容区域 */}
      {frame.visual_text && (
        <div className="visual-section">
          <h4>👁️ 视觉内容</h4>
          <pre>{frame.visual_text}</pre>
        </div>
      )}
    </div>
  );
}

function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}
```

### Vue 组件示例

```vue
<template>
  <div class="frame-card">
    <!-- 帧图片 -->
    <img
      :src="`data:image/jpeg;base64,${frame.frame_image_base64}`"
      :alt="`Frame at ${frame.video_timestamp}s`"
    />

    <!-- 时间戳 -->
    <div class="timestamp">{{ formatTime(frame.video_timestamp) }}</div>

    <!-- 语音内容 -->
    <div v-if="frame.asr_text" class="audio-section">
      <h4>🎤 语音内容</h4>
      <p>{{ frame.asr_text }}</p>
    </div>

    <!-- 视觉内容 -->
    <div v-if="frame.visual_text" class="visual-section">
      <h4>👁️ 视觉内容</h4>
      <pre>{{ frame.visual_text }}</pre>
    </div>
  </div>
</template>

<script>
export default {
  props: ['frame'],
  methods: {
    formatTime(seconds) {
      const mins = Math.floor(seconds / 60);
      const secs = Math.floor(seconds % 60);
      return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
  }
}
</script>
```

## 时间轴展示示例

```jsx
function VideoTimeline({ frames }) {
  return (
    <div className="timeline">
      {frames.map((frame, idx) => (
        <div key={idx} className="timeline-item">
          <div className="time">{formatTime(frame.video_timestamp)}</div>

          <div className="content-columns">
            {/* 左侧：语音 */}
            <div className="audio-column">
              {frame.asr_text || <span className="empty">无语音</span>}
            </div>

            {/* 中间：图片 */}
            <div className="image-column">
              <img src={`data:image/jpeg;base64,${frame.frame_image_base64}`} />
            </div>

            {/* 右侧：视觉 */}
            <div className="visual-column">
              {frame.visual_text || <span className="empty">无文字</span>}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
```

## 检索结果展示

当用户搜索时，可以高亮匹配的内容：

```jsx
function SearchResult({ frame, query }) {
  const highlightText = (text, query) => {
    if (!text || !query) return text;
    const regex = new RegExp(`(${query})`, 'gi');
    return text.replace(regex, '<mark>$1</mark>');
  };

  return (
    <div className="search-result">
      <img src={`data:image/jpeg;base64,${frame.frame_image_base64}`} />

      <div className="matched-content">
        {/* 语音匹配 */}
        {frame.asr_text?.includes(query) && (
          <div className="match-audio">
            <span className="label">🎤 语音:</span>
            <span dangerouslySetInnerHTML={{
              __html: highlightText(frame.asr_text, query)
            }} />
          </div>
        )}

        {/* 视觉匹配 */}
        {frame.visual_text?.includes(query) && (
          <div className="match-visual">
            <span className="label">👁️ 视觉:</span>
            <span dangerouslySetInnerHTML={{
              __html: highlightText(frame.visual_text, query)
            }} />
          </div>
        )}
      </div>

      <div className="timestamp">
        <a href={`#video?t=${frame.video_timestamp}`}>
          跳转到 {formatTime(frame.video_timestamp)}
        </a>
      </div>
    </div>
  );
}
```

## 说话人分离展示

如果启用了说话人分离，可以用不同颜色区分：

```jsx
function SpeakerText({ asrText }) {
  // 解析说话人标注
  const segments = asrText.split(/(\[说话人\d+\])/).filter(Boolean);

  const speakerColors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'];

  return (
    <div className="speaker-text">
      {segments.map((seg, idx) => {
        const match = seg.match(/\[说话人(\d+)\]/);
        if (match) {
          const speakerNum = parseInt(match[1]);
          return (
            <span
              key={idx}
              className="speaker-label"
              style={{ color: speakerColors[speakerNum % speakerColors.length] }}
            >
              说话人{speakerNum}:
            </span>
          );
        }
        return <span key={idx}>{seg}</span>;
      })}
    </div>
  );
}
```

## API 查询示例

### 按时间范围查询
```python
# 查询 10-20 秒之间的所有帧
frames = db.query({
    "video_filename": "demo_video.mp4",
    "video_timestamp": {"$gte": 10.0, "$lte": 20.0}
})
```

### 按内容检索
```python
# 全文检索（同时搜索语音和视觉内容）
frames = db.search({
    "content_ltks": "产品功能"
})

# 仅搜索语音内容
frames = db.search({
    "asr_text": {"$regex": "产品功能"}
})

# 仅搜索视觉内容
frames = db.search({
    "visual_text": {"$regex": "产品功能"}
})
```
