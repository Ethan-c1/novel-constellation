"""
LLM 客户端封装 - 兼容 OpenAI 格式接口
支持 DeepSeek / Qwen / GPT / SiliconFlow 等
"""
import os
import json
import logging
from openai import OpenAI

logger = logging.getLogger('silverfish.llm')


class LLMClient:
    def __init__(self):
        self.api_key = os.getenv('LLM_API_KEY', '')
        self.base_url = os.getenv('LLM_BASE_URL', 'https://api.deepseek.com/v1')
        self.model = os.getenv('LLM_MODEL_NAME', 'deepseek-chat')
        self.boost_model = os.getenv('LLM_BOOST_MODEL_NAME', '') or self.model

        if not self.api_key or self.api_key == '你的_API_KEY':
            raise ValueError(
                "LLM_API_KEY 未配置。请在项目根目录创建 .env 文件并填入 API Key。"
            )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info(f"LLM 客户端初始化: base_url={self.base_url}, model={self.model}")

    def chat_json(self, messages, temperature=0.1, use_boost=False,
                  max_tokens=8192, request_timeout=120.0):
        """
        调用 LLM 并返回解析后的 JSON。
        use_boost=True 时使用加速模型。
        """
        model = self.boost_model if use_boost else self.model

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=request_timeout,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            return self._parse_json(content)
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise

    def _parse_json(self, content):
        """容错 JSON 解析"""
        content = content.strip()
        # 去掉可能的 markdown 代码块标记
        if content.startswith('```'):
            lines = content.split('\n')
            lines = [l for l in lines if not l.strip().startswith('```')]
            content = '\n'.join(lines)

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 尝试找到第一个 { 和最后一个 }
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                return json.loads(content[start:end + 1])
            raise ValueError(f"无法解析 LLM 返回的 JSON: {content[:200]}...")
