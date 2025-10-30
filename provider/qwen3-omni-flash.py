from typing import Any

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError


class Qwen3OmniFlashProvider(ToolProvider):
    
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        try:
            api_key = credentials.get("qwen3_api_key")
            if not api_key:
                raise ToolProviderCredentialValidationError("Missing 'qwen3_api_key'")

        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))
