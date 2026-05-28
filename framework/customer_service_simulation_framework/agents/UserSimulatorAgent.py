from typing import Dict, List, Any, Optional
from agents.base.GenericAgent import GenericAgent
from core.types import Response
from core.registry import register_component
import logging
import json
import re

logger = logging.getLogger(__name__)


@register_component("agent", "user")
class UserSimulatorAgent(GenericAgent):
    def __init__(self, component_type: str, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(component_type, name, config)
    
    def _initialize_agent(self):
        try:
            self.prompt_template = self._load_prompt(self.config["prompt_file"])
            self.user_profiles = self._load_user_profiles(self.config["user_profiles_file"])
            self.core_need_file = self._load_core_needs(self.config["user_core_need_file"])
        except KeyError as e:
            raise Exception(f"User agent initialization failed, missing config: {e}")
        except Exception as e:
            raise Exception(f"User agent initialization failed: {e}")
    
    def _parse_response(self, llm_response: str) -> Response:
        try:
            is_end = False
            satisfaction = None
            original_end_tag = None
            match = re.search(r"\[end(?::(\d+))?\]", llm_response)
            if match:
                is_end = True
                original_end_tag = match.group(0)
                if match.group(1) is not None:
                    satisfaction = int(match.group(1))
                llm_response = llm_response.replace(match.group(0), "")
            
            metadata = {"role": "user", "is_end": is_end}
            if satisfaction is not None:
                metadata["satisfaction"] = satisfaction
            if original_end_tag is not None:
                metadata["original_end_tag"] = original_end_tag

            return Response(content=llm_response, success=True, metadata=metadata)

        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            return Response(content="", success=False, metadata={"role": "user", "is_end": True})
        
    def _load_user_profiles(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('user_profiles', {})
        except Exception as e:
            logger.error(f"Failed to load user profiles: {e}")
            return {}
    
    def _load_core_needs(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Failed to load core needs: {e}")
            return []
