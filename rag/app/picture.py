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

import asyncio
import base64
import io
import logging
import os
import re
import tempfile
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from api.db.services.llm_service import LLMBundle
from common.constants import LLMType
from common.string_utils import clean_markdown_block
from deepdoc.vision import OCR
from rag.nlp import attach_media_context, rag_tokenizer, tokenize
from rag.utils.asr_client import ASRClient
from rag.utils.minio_conn import RAGFlowMinio

ocr = OCR()

# Gemini supported MIME types
VIDEO_EXTS = [".mp4", ".mov", ".avi", ".flv", ".mpeg", ".mpg", ".webm", ".wmv", ".3gp", ".3gpp", ".mkv"]


class VideoFrameExtractor:
    """视频抽帧器"""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

    def extract_by_interval(self, interval_seconds: float = 1.0) -> List[Tuple[float, Image.Image]]:
        """按固定时间间隔抽帧"""
        frames = []
        frame_interval = int(self.fps * interval_seconds)

        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / self.fps
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append((timestamp, pil_image))

            frame_idx += 1

        return frames

    def extract_keyframes(self, threshold: float = 30.0) -> List[Tuple[float, Image.Image]]:
        """提取关键帧(基于帧间差异)"""
        frames = []
        prev_frame = None
        frame_idx = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                mean_diff = np.mean(diff)

                if mean_diff > threshold:
                    timestamp = frame_idx / self.fps
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append((timestamp, pil_image))
            else:
                timestamp = 0.0
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append((timestamp, pil_image))

            prev_frame = gray
            frame_idx += 1

        return frames

    def close(self):
        """释放视频资源"""
        if self.cap:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def align_asr_to_frames(asr_segments: List[Dict], frame_timestamps: List[float],
                        frame_interval: float = 1.0) -> Dict[float, str]:
    """将ASR转写文本与视频帧时间戳对齐

    使用时间段重叠算法：每帧覆盖 [timestamp, timestamp + frame_interval]，
    ASR片段覆盖 [begin_sec, end_sec]，判断两者是否有重叠。

    Args:
        asr_segments: 图灵ASR返回的分段列表，每个元素包含 begin_sec, end_sec, text, spk
        frame_timestamps: 帧时间戳列表
        frame_interval: 帧间隔（秒），用于计算每帧覆盖的时间段

    Returns:
        {帧时间戳: 对齐后的文本}
    """
    aligned = {}

    for frame_ts in frame_timestamps:
        frame_start = frame_ts
        frame_end = frame_ts + frame_interval
        matched_texts = []

        for seg in asr_segments:
            seg_start = seg.get('begin_sec', 0.0)
            seg_end = seg.get('end_sec', 0.0)

            # 时间段重叠判断: NOT (seg_end < frame_start OR seg_start > frame_end)
            if not (seg_end < frame_start or seg_start > frame_end):
                text = seg.get('text', '')
                if text:
                    spk = seg.get('spk')
                    if spk is not None:
                        text = f"[说话人{spk}] {text}"
                    matched_texts.append(text)

        aligned[frame_ts] = ' '.join(matched_texts)

    return aligned


def extract_audio_from_video(video_path: str, output_audio_path: str) -> bool:
    """从视频中提取音频"""
    try:
        import subprocess

        cmd = [
            '/ragflow/ffmpeg/ffmpeg',
            '-i', video_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            output_audio_path,
            '-y'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        logging.error(f"Failed to extract audio: {e}")
        return False


def _process_video_simple(filename, binary, tenant_id, lang, callback):
    """简单视频处理方案：直接用VLM处理整个视频（原有逻辑）"""
    doc = {
        "docnm_kwd": filename,
        "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename)),
        "doc_type_kwd": "video",
    }
    doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
    eng = lang.lower() == "english"

    try:
        cv_mdl = LLMBundle(tenant_id, llm_type=LLMType.IMAGE2TEXT, lang=lang)
        ans = asyncio.run(
            cv_mdl.async_chat(system="", history=[], gen_conf={}, video_bytes=binary, filename=filename))
        callback(0.8, "CV LLM respond: %s ..." % ans[:32])
        ans += "\n" + ans
        tokenize(doc, ans, eng)
        return [doc]
    except Exception as e:
        callback(prog=-1, msg=str(e))
        return []


def _process_video_advanced(filename, binary, tenant_id, lang, callback, parser_config):
    """高级视频处理方案：抽帧 + ASR + OCR + VLM（新设计方案）"""
    doc = {
        "docnm_kwd": filename,
        "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename)),
    }
    doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])

    # 配置参数
    frame_interval = parser_config.get('frame_interval', 1.0)
    use_keyframe = parser_config.get('use_keyframe', False)
    enable_asr = parser_config.get('enable_asr', True)
    enable_ocr = parser_config.get('enable_ocr', True)
    enable_vlm = parser_config.get('enable_vlm', False)
    asr_type = parser_config.get('asr_type', 'whisper')  # 'whisper' 或 'tuling'
    asr_api_url = parser_config.get('asr_api_url', '')
    asr_language = parser_config.get('asr_language', 'auto')  # Whisper语言代码
    speaker_separation = parser_config.get('speaker_separation', False)

    video_path = None
    audio_path = None
    chunks = []

    try:
        _, ext = os.path.splitext(filename)
        if not ext:
            raise RuntimeError("No extension detected.")

        # 保存视频到临时文件
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmpf:
            tmpf.write(binary)
            tmpf.flush()
            video_path = os.path.abspath(tmpf.name)

        callback(0.1, "开始提取视频帧...")

        # 提取视频帧
        with VideoFrameExtractor(video_path) as extractor:
            video_duration = extractor.duration
            video_fps = extractor.fps
            if use_keyframe:
                frames = extractor.extract_keyframes()
            else:
                frames = extractor.extract_by_interval(frame_interval)

        callback(0.3, f"提取了 {len(frames)} 个视频帧")

        # ASR转写
        asr_segments = []
        if enable_asr:
            callback(0.4, "开始音频转写...")

            audio_path = video_path.replace(ext, '.wav')
            if extract_audio_from_video(video_path, audio_path):
                try:
                    # 读取音频字节数组
                    with open(audio_path, 'rb') as f:
                        audio_bytes = f.read()

                    # 初始化ASR客户端
                    asr_client = ASRClient(
                        asr_type=asr_type,
                        api_url=asr_api_url if asr_api_url else None,
                        timeout=300
                    )

                    # 调用ASR转写
                    asr_segments = asr_client.transcribe(
                        audio_bytes=audio_bytes,
                        speaker_separation=speaker_separation,
                        language=asr_language,
                    )

                    if asr_segments:
                        preview = asr_segments[0].get('text', '')[:50]
                        callback(0.6, f"音频转写完成({len(asr_segments)}段): {preview}...")
                    else:
                        callback(0.6, "音频转写完成，未识别到语音内容")

                except Exception as e:
                    logging.error(f"ASR failed: {e}")
                    callback(0.6, f"音频转写失败: {str(e)}")
            else:
                callback(0.6, "音频提取失败，跳过ASR转写")

        # 对齐ASR文本到帧
        frame_timestamps = [ts for ts, _ in frames]
        aligned_texts = align_asr_to_frames(asr_segments, frame_timestamps, frame_interval) if asr_segments else {}

        callback(0.7, "开始处理视频帧...")

        # 预先初始化OCR引擎
        ocr_engine = None
        if enable_ocr:
            try:
                ocr_engine = OCR()
            except Exception as e:
                logging.error(f"Failed to initialize OCR engine: {e}")

        # 预先初始化VLM模型
        vision_model = None
        llm_id = parser_config.get('llm_id', '')
        if enable_vlm and llm_id:
            try:
                vision_model = LLMBundle(tenant_id, LLMType.IMAGE2TEXT, llm_name=llm_id)
            except Exception as e:
                logging.error(f"Failed to initialize VLM model: {e}")

        # 处理每一帧
        for idx, (timestamp, frame_image) in enumerate(frames):
            frame_doc = doc.copy()

            # 添加视频特有字段
            frame_doc['video_timestamp'] = timestamp
            frame_doc['frame_index'] = idx
            frame_doc['is_video_frame'] = True
            frame_doc['video_filename'] = filename
            frame_doc['video_duration'] = video_duration
            frame_doc['video_fps'] = video_fps

            # 获取对齐的ASR文本
            asr_text = aligned_texts.get(timestamp, '')

            # OCR处理
            ocr_text = ''
            if ocr_engine:
                try:
                    bxs = ocr_engine(np.array(frame_image))
                    ocr_text = "\n".join([t[0] for _, t in bxs if t[0]])
                except Exception as e:
                    logging.error(f"OCR failed for frame {idx}: {e}")

            # VLM处理
            vlm_text = ''
            if vision_model:
                try:
                    vlm_text = vision_llm_chunk(frame_image, vision_model, callback=callback)
                except Exception as e:
                    logging.error(f"VLM failed for frame {idx}: {e}")

            # 融合多模态内容
            combined_text = '\n'.join(filter(None, [asr_text, ocr_text, vlm_text]))

            if combined_text:
                is_english = lang.lower() == "english"
                tokenize(frame_doc, combined_text, is_english)

                # 存储帧图片到MinIO
                try:
                    minio_client = RAGFlowMinio()

                    img_buffer = io.BytesIO()
                    frame_image.save(img_buffer, format='JPEG', quality=85)
                    img_bytes = img_buffer.getvalue()

                    filename_base = os.path.splitext(filename)[0]
                    frame_filename = f"videos/{tenant_id}/{filename_base}/frame_{idx}_{timestamp:.2f}.jpg"

                    minio_client.put(
                        bucket=tenant_id,
                        fnm=frame_filename,
                        binary=img_bytes,
                        tenant_id=tenant_id
                    )

                    frame_doc['frame_image_path'] = frame_filename
                    frame_doc['frame_image_base64'] = base64.b64encode(img_bytes).decode('utf-8')

                except Exception as e:
                    logging.error(f"Failed to upload frame {idx} to MinIO: {e}")

                chunks.append(frame_doc)

        callback(1.0, f"视频处理完成,生成 {len(chunks)} 个chunks")
        return chunks

    except Exception as e:
        logging.exception(f"Video processing failed: {e}")
        callback(prog=-1, msg=str(e))
        return []
    finally:
        if video_path and os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except Exception as e:
                logging.exception(f"Failed to remove temporary video file: {video_path}, exception: {e}")

        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except Exception as e:
                logging.exception(f"Failed to remove temporary audio file: {audio_path}, exception: {e}")


def chunk(filename, binary, tenant_id, lang, callback=None, **kwargs):
    doc = {
        "docnm_kwd": filename,
        "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename)),
    }
    eng = lang.lower() == "english"

    parser_config = kwargs.get("parser_config", {}) or {}
    image_ctx = max(0, int(parser_config.get("image_context_size", 0) or 0))

    # 检查是否是视频文件
    if any(filename.lower().endswith(ext) for ext in VIDEO_EXTS):
        # 通过配置开关选择视频处理方案
        # use_advanced_video_processing: True=高级方案(抽帧+ASR+OCR), False=简单方案(直接VLM)
        use_advanced = parser_config.get('use_advanced_video_processing', True)

        if use_advanced:
            return _process_video_advanced(filename, binary, tenant_id, lang, callback, parser_config)
        else:
            return _process_video_simple(filename, binary, tenant_id, lang, callback)
    else:
        # 图片处理逻辑（保持不变）
        img = Image.open(io.BytesIO(binary)).convert("RGB")
        doc.update(
            {
                "image": img,
                "doc_type_kwd": "image",
            }
        )
        bxs = ocr(np.array(img))
        txt = "\n".join([t[0] for _, t in bxs if t[0]])
        callback(0.4, "Finish OCR: (%s ...)" % txt[:12])
        if (eng and len(txt.split()) > 32) or len(txt) > 32:
            tokenize(doc, txt, eng)
            callback(0.8, "OCR results is too long to use CV LLM.")
            return attach_media_context([doc], 0, image_ctx)

        try:
            callback(0.4, "Use CV LLM to describe the picture.")
            cv_mdl = LLMBundle(tenant_id, LLMType.IMAGE2TEXT, lang=lang)
            with io.BytesIO() as img_binary:
                img.save(img_binary, format="JPEG")
                img_binary.seek(0)
                ans = cv_mdl.describe(img_binary.read())
            callback(0.8, "CV LLM respond: %s ..." % ans[:32])
            txt += "\n" + ans
            tokenize(doc, txt, eng)
            return attach_media_context([doc], 0, image_ctx)
        except Exception as e:
            callback(prog=-1, msg=str(e))

    return []


def vision_llm_chunk(binary, vision_model, prompt=None, callback=None):
    """
    A simple wrapper to process image to markdown texts via VLM.

    Returns:
        Simple markdown texts generated by VLM.
    """
    callback = callback or (lambda prog, msg: None)

    img = binary
    txt = ""

    try:
        # Skip tiny crops that fail provider image-size limits.
        if hasattr(img, "size"):
            min_side = 11
            if img.size[0] < min_side or img.size[1] < min_side:
                callback(0.0, f"Skip tiny image for VLM: {img.size[0]}x{img.size[1]}")
                return ""
        with io.BytesIO() as img_binary:
            try:
                img.save(img_binary, format="JPEG")
            except Exception:
                img_binary.seek(0)
                img_binary.truncate()
                img.save(img_binary, format="PNG")

            img_binary.seek(0)
            ans = clean_markdown_block(vision_model.describe_with_prompt(img_binary.read(), prompt))
            txt += "\n" + ans
            return txt

    except Exception as e:
        callback(-1, str(e))

    return ""
