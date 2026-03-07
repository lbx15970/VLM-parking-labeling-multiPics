import concurrent.futures
import os
import base64
import requests
from openai import OpenAI


class QwenRunner:
    def __init__(self, api_key, model_id, base_url=None, enable_thinking=True):
        self.api_key = api_key
        self.model_id = model_id
        self.enable_thinking = enable_thinking
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = OpenAI(base_url=self.base_url, api_key=api_key)
        self._image_cache = {}  # cache base64 by URL

    def _get_image_base64(self, image_path):
        """将图片转为 base64 data URL。支持本地文件路径和远程 URL。"""
        if image_path in self._image_cache:
            return self._image_cache[image_path]

        if os.path.isfile(image_path):
            with open(image_path, 'rb') as f:
                image_data = f.read()
            image_name = os.path.basename(image_path)
        else:
            resp = requests.get(image_path, timeout=30)
            resp.raise_for_status()
            image_data = resp.content
            image_name = os.path.basename(image_path)

        b64 = base64.b64encode(image_data).decode('utf-8')
        ext = os.path.splitext(image_name)[1].lower()
        mime_map = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}
        mime = mime_map.get(ext, 'image/jpeg')
        data_url = f"data:{mime};base64,{b64}"

        self._image_cache[image_path] = data_url
        return data_url

    def _invoke(self, prompt_text, image_url):
        # Convert image to base64 data URL since Qwen may not access TOS URLs
        data_url = self._get_image_base64(image_url)

        extra_body = {"vl_high_resolution_images": True}
        if self.enable_thinking:
            extra_body["enable_thinking"] = True

        return self.client.chat.completions.create(
            model=self.model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }],
            extra_body=extra_body,
            temperature=0.1,
            top_p=0.7
        )

    def run(self, prompt_text, image_url, timeout_seconds):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._invoke, prompt_text, image_url)
            return future.result(timeout=timeout_seconds)
