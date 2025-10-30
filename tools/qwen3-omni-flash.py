import json
from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from openai import OpenAI

class Qwen3OmniFlashTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:

        # 初始化OpenAI客户端
        client = OpenAI(
            # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
            api_key=self.runtime.credentials["qwen3_api_key"],
            # 以下是北京地域base_url，如果使用新加坡地域的模型，需要将base_url替换为：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
            base_url=self.runtime.credentials.get("qwen3_api_url") or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        system_text = tool_parameters.get("system_text")
        user_text = tool_parameters.get("user_text")
        user_audio_base64 = tool_parameters.get("user_audio_base64")
        context = tool_parameters.get("context")

        if not system_text and not user_text and not user_audio_base64 and not context:
            yield self.create_json_message({"status": "error", "error": "No valid message provided"})

        # 构建请求消息
        messages = []

        if context:
            try:
                messages = json.loads(context)
            except json.JSONDecodeError:
                yield self.create_json_message({"status": "error", "error": "Context is not a valid JSON string"})

        if system_text:
            system_text_exists = False

            for i in range(len(messages)):
                if messages[i].get("role") == "system":
                    messages[i]["content"]["text"] = system_text
                    system_text_exists = True
                    break

            if not system_text_exists:
                messages.insert(0, {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{system_text}"
                        }
                    ]
                })

        user_query = []

        if user_text:
            user_query.append({
                "type": "text",
                "text": user_text
            })

        if user_audio_base64:
            user_query.append({
                "type": "input_audio",
                "input_audio": {
                    "data": f"data:;base64,{user_audio_base64}",
                    "format": "wav"
                }
            })

        if user_query:
            messages.append({
                "role": "user",
                "content": user_query
            })

        yield self.create_json_message({"messages": messages})

        completion = client.chat.completions.create(
            model="qwen3-omni-flash",
            messages=messages,
            modalities=["text", "audio"],
            audio={"voice": "Cherry", "format": "wav"},
            stream=True, # stream 必须设置为 True，否则会报错
            stream_options={"include_usage": False},
        )

        assistant_reply_text = ""
        assistant_reply_audio_base64 = ""

        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                assistant_reply_text += chunk.choices[0].delta.content

            if chunk.choices and hasattr(chunk.choices[0].delta, "audio") and chunk.choices[0].delta.audio:
                assistant_reply_audio_base64 += chunk.choices[0].delta.audio.get("data", "")

        messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"{assistant_reply_text}"
                }
            ]
        })

        yield self.create_text_message(assistant_reply_text)


        # yield self.create_variable_message("audio_base64", assistant_reply_audio_base64)
        # yield self.create_variable_message("messages", messages)

        yield self.create_json_message({
            "status": "success",
            "response": {
                "text": assistant_reply_text,
                "audio_base64": assistant_reply_audio_base64,
            },
            "messages": messages
        })
