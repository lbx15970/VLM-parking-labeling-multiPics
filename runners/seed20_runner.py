import concurrent.futures
import os
import base64
from volcenginesdkarkruntime import Ark


class Seed20Runner:
    def __init__(self, api_key, ep, model_id, base_url):
        self.api_key = api_key
        self.ep = ep or os.getenv("SEED20_EP", "").strip()
        self.model_id = model_id
        self.base_url = base_url
        self.client = Ark(base_url=base_url, api_key=api_key)

    def _resolve_image_url(self, image_path):
        """本地文件转 base64 data URL，远程 URL 直接返回"""
        if os.path.isfile(image_path):
            with open(image_path, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
            ext = os.path.splitext(image_path)[1].lower()
            mime_map = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}
            mime = mime_map.get(ext, 'image/jpeg')
            return f"data:{mime};base64,{b64}"
        return image_path

    def _invoke(self, prompt_text, image_path):
        image_url = self._resolve_image_url(image_path)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_url}, "detail": "high"}
            ]
        }]
        try:
            return self.client.chat.completions.create(
                model=self.ep or self.model_id,
                messages=messages,
                extra_body={"thinking": {"type": "enabled", "reasoning": {"effort": "high"}}},
                temperature=0.1,
                top_p=0.7
            )
        except Exception as e:
            if self.ep and "AccessDenied" in str(e):
                try:
                    return self.client.chat.completions.create(
                        model=self.model_id,
                        messages=messages,
                        extra_body={"thinking": {"type": "enabled", "reasoning": {"effort": "high"}}},
                        temperature=0.1,
                        top_p=0.7
                    )
                except Exception:
                    raise e
            raise

    def run(self, prompt_text, image_path, timeout_seconds):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._invoke, prompt_text, image_path)
            return future.result(timeout=timeout_seconds)

