from typing import Dict, Any, Optional
from agents.base.GenericAgent import GenericAgent
from core.types import Response
from core.registry import register_component
import logging
import re
import json

logger = logging.getLogger(__name__)


@register_component("agent", "customer")
class CustomerServiceAgent(GenericAgent):
    def __init__(self, component_type: str, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(component_type, name, config)
    
    def _initialize_agent(self):
        try:
            self.prompt_template = self._load_prompt(self.config["prompt_file"])
        except KeyError as e:
            raise Exception(f"Customer service agent initialization failed, missing config: {e}")
        except Exception as e:
            raise Exception(f"Customer service agent initialization failed: {e}")

    def _parse_response(self, llm_response: str) -> Response:
        try:
            think_match = re.search(r'<think>(.*?)</think>', llm_response, re.DOTALL)
            think_content = think_match.group(1).strip() if think_match else ""

            response_match = re.search(r'<response>(.*?)</response>', llm_response, re.DOTALL)
            response_content = response_match.group(1).strip() if response_match else ""

            # First try to extract the <action> tag
            action_match = re.search(r'<action>(.*?)</action>', llm_response, re.DOTALL)
            action_content = action_match.group(1).strip().lower() if action_match else None

            # If <action> was not extracted, try to extract from <tool_calls> JSON
            if action_content is None:
            # if action_content:
                tool_calls_match = re.search(r'<tool>(.*?)</tool>', llm_response, re.DOTALL)
                # tool_calls_match = re.search(r'<action>(.*?)</action>', llm_response, re.DOTALL)
                if tool_calls_match:
                    try:
                        tool_calls_str = tool_calls_match.group(1).strip()
                        tool_calls_json = json.loads(tool_calls_str)
                        if isinstance(tool_calls_json, dict) and 'arguments' in tool_calls_json:
                            action_content = tool_calls_json['arguments'].get('action', 'chat').lower()
                        elif isinstance(tool_calls_json, list) and len(tool_calls_json) > 0:
                            action_content = tool_calls_json[0].get('arguments', {}).get('action', 'chat').lower()
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        logger.warning(f"Failed to parse <tool_call> JSON: {e}")
                        action_content = 'chat'
                else:
                    action_content = 'chat'

            if action_content not in ['chat', 'voucher','transfer']:
                action_content = 'chat'

            if action_content == 'voucher' and response_content:
                response_content += "[SYSTEM: A coupon has been issued to you.]"
            
            metadata = {"role": "assistant", "think": think_content, "action": action_content, "full_response": llm_response}

            if action_content == 'transfer':
                metadata['is_end'] = True

            return Response(
                content=response_content,
                success=True,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Failed to parse XML response: {e}")
            default = self._get_default_response("Response format parsing failed")
            return Response(
                content=default.get("response", ""),
                success=False,
                metadata={"role": "assistant", "think": default.get("think"), "action": default.get("action", "chat"), "full_response": llm_response}
            )
    
    def _get_default_response(self, reason: str) -> Dict[str, Any]:
        return {
            'think': f"System error: {reason}",
            'response': "Sorry, I encountered a technical issue. Please try again later.",
            'action': 'chat'
        }
    