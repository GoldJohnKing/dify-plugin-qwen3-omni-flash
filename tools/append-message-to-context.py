import json
from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage


class AppendMessageToContextTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:

        role = tool_parameters["role"]
        message = tool_parameters["message"]
        context = tool_parameters.get("context")

        new_context = []

        if context and context.strip():
            try:
                new_context = json.loads(context)
            except json.JSONDecodeError:
                yield self.create_json_message({"status": "error", "error": "Context is not a valid JSON string"})

        if message and message.strip():
            new_context.append({
                "role": role,
                "content": [
                    {
                        "type": "text",
                        "text": message
                    }
                ]
            })
        else:
            yield self.create_json_message({"status": "error", "error": "Message is not a valid string"})

        new_context_string = json.dumps(new_context, ensure_ascii=False)

        yield self.create_text_message(new_context_string)

        yield self.create_json_message({
            "status": "success",
            "context": new_context_string
        })
