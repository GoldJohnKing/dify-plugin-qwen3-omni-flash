import json
from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from openai import OpenAI

class Qwen3OmniFlashTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:

        client = OpenAI(
            api_key=self.runtime.credentials["qwen3_api_key"],
            base_url=self.runtime.credentials.get("qwen3_api_url") or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        modal_type = tool_parameters["modal_type"]
        response_modal_type = tool_parameters["response_modal_type"]

        context = tool_parameters.get("context")
        system_prompt = tool_parameters.get("system_prompt")
        user_query_text = tool_parameters.get("user_query_text")

        if modal_type != "text":
            modal_payload_type = tool_parameters.get("modal_payload_type")
            modal_payload = tool_parameters.get("modal_payload")

        # Check if no valid message
        if not system_prompt and not user_query_text and not context and (modal_type == "text" or not modal_payload):
            yield self.create_json_message({"status": "error", "error": "No valid message provided"})

        # Construct message
        messages = []

        if context:
            try:
                messages = json.loads(context)
            except json.JSONDecodeError:
                yield self.create_json_message({"status": "error", "error": "Context is not a valid JSON string"})

        if system_prompt:
            if messages and messages[0].get("role") == "system":
                messages.pop(0)
            messages.insert(0, {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"{system_prompt}"
                    }
                ]
            })

        user_query = []

        if user_query_text:
            user_query.append({
                "type": "text",
                "text": user_query_text
            })

        if modal_type == "audio" and modal_payload:
            audio_format = tool_parameters.get("audio_format")
            if not audio_format:
                yield self.create_json_message({"status": "error", "error": "Audio modality format is required"})

            payload_data = f"data:;base64,{modal_payload}" if modal_payload_type == "base64" else modal_payload

            user_query.append({
                "type": "input_audio",
                "input_audio": {
                    "data": payload_data,
                    "format": audio_format
                }
            })

        # if modal_type == "image" and modal_payload:
        #     image_format = tool_parameters.get("audio_format")
        #     if not audio_format:
        #         yield self.create_json_message({"status": "error", "error": "Audio modality format is required"})
        #
        #     payload_data = f"data:image/png;base64,{modal_payload}" if modal_payload_type == "base64" else modal_payload
        #
        #     user_query.append({
        #         "type": "image_url",
        #         "image_url": {
        #             "url": payload_data
        #         }
        #     })

        # if modal_type == "video" and modal_payload:
        #     type = "video_url"

        # if modal_type == "video_image_sequence" and modal_payload:
        #     type = "video"

        if user_query:
            messages.append({
                "role": "user",
                "content": user_query
            })

        modalities = ["text"]

        if response_modal_type == "audio":
            modalities.append("audio")

        completion = client.chat.completions.create(
            model="qwen3-omni-flash",
            messages=messages,
            modalities=modalities,
            audio={"voice": "Cherry", "format": "wav"},
            stream=True,
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

        yield self.create_variable_message("context", json.dumps(messages))

        yield self.create_json_message({
            "status": "success",
            "response": {
                "text": assistant_reply_text,
                "audio_base64": assistant_reply_audio_base64,
            },
            "context": messages
        })
