#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
from typing import Dict, List, Optional, Union

import requests


# 图灵ASR默认帧率（每秒帧数），用于将返回的帧数转换为秒
DEFAULT_TULING_FRAME_RATE = 100.0

# 默认ASR服务地址
DEFAULT_TULING_ASR_URL = "http://112.39.27.207:8900/tuling/asr/v3/process"
DEFAULT_WHISPER_ASR_URL = "http://172.31.169.57:9001/asr"


class ASRClient:
    """统一的ASR客户端接口，支持图灵ASR和Whisper"""

    def __init__(self, asr_type: str = "whisper", api_url: Optional[str] = None, timeout: int = 300, **kwargs):
        """
        Args:
            asr_type: ASR类型，"tuling" 或 "whisper"
            api_url: ASR服务地址，不指定则使用默认值
            timeout: 请求超时时间（秒）
            **kwargs: 其他参数，如 frame_rate（图灵ASR专用）
        """
        self.asr_type = asr_type.lower()
        self.timeout = timeout

        if self.asr_type == "tuling":
            self.api_url = api_url or DEFAULT_TULING_ASR_URL
            self.frame_rate = kwargs.get("frame_rate", DEFAULT_TULING_FRAME_RATE)
        elif self.asr_type == "whisper":
            self.api_url = api_url or DEFAULT_WHISPER_ASR_URL
        else:
            raise ValueError(f"Unsupported ASR type: {asr_type}, must be 'tuling' or 'whisper'")

    def transcribe(self, audio_bytes: bytes, speaker_separation: bool = False,
                   language: str = "auto", **kwargs) -> List[Dict]:
        """
        调用ASR服务进行转写

        Args:
            audio_bytes: 音频文件的字节数组（WAV格式）
            speaker_separation: 是否进行说话人分离
            language: 语言代码（Whisper用），如 "zh", "en", "auto"
            **kwargs: 其他参数

        Returns:
            转写结果列表，每个元素包含:
            - begin_sec: 开始时间（秒）
            - end_sec: 结束时间（秒）
            - text: 转写文本
            - spk: 说话人序号（仅在speaker_separation=True时有值）
            - confidence: 置信度（可选）
        """
        if self.asr_type == "tuling":
            return self._transcribe_tuling(audio_bytes, speaker_separation, **kwargs)
        elif self.asr_type == "whisper":
            return self._transcribe_whisper(audio_bytes, speaker_separation, language, **kwargs)

    def _transcribe_tuling(self, audio_bytes: bytes, speaker_separation: bool, **kwargs) -> List[Dict]:
        """图灵ASR转写（暂时返回模拟数据）"""
        # TODO: 图灵ASR服务尚未部署，暂时返回模拟数据
        logging.warning("Tuling ASR service is not deployed yet, returning mock data.")
        return [
            {"begin_sec": 0.0, "end_sec": 3.0, "text": "大家好，欢迎观看本视频", "spk": 0, "confidence": 0.95},
            {"begin_sec": 3.0, "end_sec": 6.5, "text": "今天我们来介绍一下这个产品的主要功能", "spk": 0, "confidence": 0.92},
            {"begin_sec": 6.5, "end_sec": 10.0, "text": "首先我们看一下它的界面设计", "spk": 0, "confidence": 0.90},
            {"begin_sec": 10.0, "end_sec": 15.0, "text": "这里可以看到整体的布局非常简洁", "spk": 0, "confidence": 0.88},
            {"begin_sec": 15.0, "end_sec": 20.0, "text": "接下来我们演示一下具体的操作流程", "spk": 0, "confidence": 0.91},
        ]

        # 图灵ASR需要HTTP可访问的URI，需要先上传到MinIO或其他存储
        # 这里暂时注释掉，等服务部署后再启用
        # audio_id = kwargs.get("audio_id") or str(uuid.uuid4())
        # audio_uri = kwargs.get("audio_uri")  # 需要外部提供
        #
        # if not audio_uri:
        #     raise ValueError("Tuling ASR requires audio_uri parameter")
        #
        # request_data = {
        #     "audio": {
        #         "aid": audio_id,
        #         "uri": audio_uri,
        #         "fmt": "wav",
        #         "spkn": 2 if speaker_separation else 1,
        #     }
        # }
        #
        # try:
        #     logging.info(f"Calling Tuling ASR: {self.api_url}, audio_uri: {audio_uri}")
        #     resp = requests.post(self.api_url, json=request_data, timeout=self.timeout)
        #     resp.raise_for_status()
        #     result = resp.json()
        #     return self._parse_tuling_response(result)
        # except requests.RequestException as e:
        #     logging.error(f"Tuling ASR request failed: {e}")
        #     return []

    def _transcribe_whisper(self, audio_bytes: bytes, speaker_separation: bool,
                            language: str, **kwargs) -> List[Dict]:
        """Whisper ASR转写"""
        try:
            # 构建请求参数
            params = {
                "output": "json",
                "task": "transcribe",
                "encode": "true",
            }

            # 语言设置
            if language and language != "auto":
                params["language"] = language

            # 说话人分离（WhisperX专用）
            if speaker_separation:
                params["diarize"] = "true"
                if "min_speakers" in kwargs:
                    params["min_speakers"] = kwargs["min_speakers"]
                if "max_speakers" in kwargs:
                    params["max_speakers"] = kwargs["max_speakers"]

            # 词级时间戳（Faster Whisper专用）
            if kwargs.get("word_timestamps"):
                params["word_timestamps"] = "true"

            # VAD过滤（Faster Whisper专用）
            if kwargs.get("vad_filter"):
                params["vad_filter"] = "true"

            # 构建multipart/form-data请求
            files = {"audio_file": ("audio.wav", audio_bytes, "audio/wav")}

            logging.info(f"Calling Whisper ASR: {self.api_url}, params: {params}")
            resp = requests.post(self.api_url, params=params, files=files, timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()

            return self._parse_whisper_response(result, speaker_separation)

        except requests.RequestException as e:
            logging.error(f"Whisper ASR request failed: {e}")
            return []

    def _parse_tuling_response(self, result: dict) -> List[Dict]:
        """解析图灵ASR返回结果"""
        segments = []

        if result.get("code") != 0:
            logging.error(f"Tuling ASR returned error: code={result.get('code')}, msg={result.get('msg')}")
            return []

        lattices = result.get("data", {}).get("lattice", [])
        if not lattices:
            logging.warning("Tuling ASR returned empty lattice")
            return []

        for lattice in lattices:
            begin_frame = lattice.get("begin", 0)
            end_frame = lattice.get("end", 0)
            text = lattice.get("text", "").strip()
            spk = lattice.get("spk")
            confidence = lattice.get("confidence", 0.0)

            if not text:
                continue

            segments.append({
                "begin_sec": begin_frame / self.frame_rate,
                "end_sec": end_frame / self.frame_rate,
                "text": text,
                "spk": spk,
                "confidence": confidence,
            })

        logging.info(f"Tuling ASR parsed {len(segments)} segments")
        return segments

    def _parse_whisper_response(self, result: dict, speaker_separation: bool) -> List[Dict]:
        """解析Whisper ASR返回结果"""
        segments = []

        whisper_segments = result.get("segments", [])
        if not whisper_segments:
            logging.warning("Whisper ASR returned empty segments")
            return []

        for seg in whisper_segments:
            text = seg.get("text", "").strip()
            if not text:
                continue

            segment = {
                "begin_sec": seg.get("start", 0.0),
                "end_sec": seg.get("end", 0.0),
                "text": text,
                "confidence": seg.get("confidence"),
            }

            # 说话人信息（WhisperX diarization）
            if speaker_separation and "speaker" in seg:
                segment["spk"] = seg["speaker"]

            segments.append(segment)

        logging.info(f"Whisper ASR parsed {len(segments)} segments")
        return segments

