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
import os
import subprocess
import uuid
from typing import Dict, List, Optional

import requests


# 图灵ASR默认帧率（每秒帧数），用于将返回的帧数转换为秒
DEFAULT_FRAME_RATE = 100.0

# 默认ASR服务地址
DEFAULT_ASR_API_URL = "http://112.39.27.207:8900/tuling/asr/v3/process"


class TulingASRClient:
    """图灵ASR转写客户端

    调用图灵ASR HTTP接口，将音频文件转写为带时间戳的文本片段。
    """

    def __init__(self, api_url: str = DEFAULT_ASR_API_URL, timeout: int = 300,
                 frame_rate: float = DEFAULT_FRAME_RATE):
        self.api_url = api_url
        self.timeout = timeout
        self.frame_rate = frame_rate

    def transcribe(self, audio_uri: str, audio_id: Optional[str] = None,
                   speaker_separation: bool = False) -> List[Dict]:
        """
        调用图灵ASR服务进行转写

        Args:
            audio_uri: 音频文件的HTTP可访问URI
            audio_id: 音频唯一标识，默认自动生成
            speaker_separation: 是否进行说话人分离

        Returns:
            转写结果列表，每个元素包含:
            - begin_sec: 开始时间（秒）
            - end_sec: 结束时间（秒）
            - text: 转写文本
            - spk: 说话人序号（仅在speaker_separation=True时有值）
            - confidence: 置信度
        """
        # TODO: ASR服务尚未部署，暂时返回模拟数据。部署后取消下方注释即可启用。
        logging.warning("Tuling ASR service is not deployed yet, returning mock data.")
        return [
            {"begin_sec": 0.0, "end_sec": 3.0, "text": "大家好，欢迎观看本视频", "spk": 0, "confidence": 0.95},
            {"begin_sec": 3.0, "end_sec": 6.5, "text": "今天我们来介绍一下这个产品的主要功能", "spk": 0, "confidence": 0.92},
            {"begin_sec": 6.5, "end_sec": 10.0, "text": "首先我们看一下它的界面设计", "spk": 0, "confidence": 0.90},
            {"begin_sec": 10.0, "end_sec": 15.0, "text": "这里可以看到整体的布局非常简洁", "spk": 0, "confidence": 0.88},
            {"begin_sec": 15.0, "end_sec": 20.0, "text": "接下来我们演示一下具体的操作流程", "spk": 0, "confidence": 0.91},
        ]

        # if not audio_id:
        #     audio_id = str(uuid.uuid4())
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
        # except requests.RequestException as e:
        #     logging.error(f"Tuling ASR request failed: {e}")
        #     return []
        #
        # return self._parse_response(result)

    def _parse_response(self, result: dict) -> List[Dict]:
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


def convert_to_wav(input_path: str, output_path: Optional[str] = None) -> Optional[str]:
    """将音频文件转换为WAV格式（16kHz, 单声道, PCM 16bit）

    Args:
        input_path: 输入音频文件路径
        output_path: 输出WAV文件路径，默认在同目录下生成

    Returns:
        输出文件路径，失败返回None
    """
    if not output_path:
        base, _ = os.path.splitext(input_path)
        output_path = base + ".wav"

    try:
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            output_path,
            "-y",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            return output_path
        logging.error(f"ffmpeg conversion failed: {result.stderr}")
        return None
    except Exception as e:
        logging.error(f"Audio conversion failed: {e}")
        return None
