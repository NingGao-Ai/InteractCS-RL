from typing import Dict, Any, Optional
from agents.base.GenericAgent import GenericAgent
from core.types import Response
from core.registry import register_component
import logging
import json
import re

logger = logging.getLogger(__name__)


@register_component("agent", "speech")
class SpeechEvaluationAgent(GenericAgent):
    """Speech Evaluation Agent - responsible for parsing evaluation results"""
    
    def __init__(self, component_type: str, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(component_type, name, config)
    
    def _initialize_agent(self):
        """Initialize speech evaluation agent"""
        try:
            self.prompt_template = self._load_prompt(self.config["prompt_file"])
            logger.debug(f"Speech evaluation agent initialized successfully")
        except KeyError as e:
            raise Exception(f"Speech evaluation agent initialization failed, missing config: {e}")
        except Exception as e:
            raise Exception(f"Speech evaluation agent initialization failed: {e}")

    def _parse_response(self, llm_response: str) -> Response:
        """
        Parse LLM evaluation results

        Expected JSON format:
        {
            "identity_neutrality": {
                "reason": "evaluation reason",
                "identity_claim_avoidance": 2,
                "alternative_expression": 1
            },
            ...
        }
        """
        try:
            # Try to extract JSON content
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                evaluation_data = json.loads(json_str)
                
                return Response(
                    content=json.dumps(evaluation_data, ensure_ascii=False),
                    success=True,
                    metadata={
                        "role": "evaluator",
                        "evaluation_type": "speech",
                        "evaluation_data": evaluation_data,
                        "evaluation_success": True
                    }
                )
            else:
                logger.error("Unable to extract JSON-formatted evaluation results from response")
                return self._get_default_response("Unable to extract JSON")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse evaluation JSON: {e}")
            logger.debug(f"Raw response: {llm_response}")
            return self._get_default_response(f"JSON parsing failed: {e}")
        except Exception as e:
            logger.error(f"Failed to parse evaluation response: {e}")
            return self._get_default_response(f"Parsing failed: {e}")

    def _get_default_response(self, reason: str) -> Response:
        """Get default evaluation result (all zeros)"""
        default_data = {
            "identity_neutrality": {"identity_claim_avoidance": 0, "alternative_expression": 0, "reason": f"Evaluation failed: {reason}"},
            "dialogue_quality": {"content_novelty": 0, "problem_focus": 0, "reason": f"Evaluation failed: {reason}"},
            "language_adaptability": {"language_style_matching": 0, "technical_complexity_adaptation": 0, "reason": f"Evaluation failed: {reason}"},
            "content_quality": {"content_substantiveness": 0, "promise_boundary_control": 0, "information_authenticity": 0, "reason": f"Evaluation failed: {reason}"},
            "communication_effectiveness": {"conciseness": 0, "sincerity": 0, "reason": f"Evaluation failed: {reason}"},
            "natural_fluency": {"sentence_diversity": 0, "transition_naturalness": 0, "reason": f"Evaluation failed: {reason}"},
            "context_adaptability": {"problem_complexity_adaptation": 0, "user_knowledge_adaptation": 0, "reason": f"Evaluation failed: {reason}"}
        }
        
        return Response(
            content=json.dumps(default_data, ensure_ascii=False),
            success=False,
            metadata={
                "role": "evaluator",
                "evaluation_type": "speech",
                "evaluation_data": default_data,
                "evaluation_success": False,
                "error": reason
            }
        )


@register_component("agent", "logic")
class LogicEvaluationAgent(GenericAgent):
    """Logic Evaluation Agent - responsible for parsing evaluation results"""
    
    def __init__(self, component_type: str, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(component_type, name, config)
    
    def _initialize_agent(self):
        """Initialize logic evaluation agent"""
        try:
            self.prompt_template = self._load_prompt(self.config["prompt_file"])
            logger.debug(f"Logic evaluation agent initialized successfully")
        except KeyError as e:
            raise Exception(f"Logic evaluation agent initialization failed, missing config: {e}")
        except Exception as e:
            raise Exception(f"Logic evaluation agent initialization failed: {e}")

    def _parse_response(self, llm_response: str) -> Response:
        """Parse LLM logic evaluation results"""
        try:
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                evaluation_data = json.loads(json_str)
                
                return Response(
                    content=json.dumps(evaluation_data, ensure_ascii=False),
                    success=True,
                    metadata={
                        "role": "evaluator",
                        "evaluation_type": "logic",
                        "evaluation_data": evaluation_data,
                        "evaluation_success": True
                    }
                )
            else:
                logger.error("Unable to extract JSON-formatted evaluation results from response")
                return self._get_default_response("Unable to extract JSON")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse evaluation JSON: {e}")
            return self._get_default_response(f"JSON parsing failed: {e}")
        except Exception as e:
            logger.error(f"Failed to parse evaluation response: {e}")
            return self._get_default_response(f"Parsing failed: {e}")

    def _get_default_response(self, reason: str) -> Response:
        """Get default logic evaluation result (all zeros)"""
        default_data = {
            "user_profile_recognition": {"user_profile_emotion_judgment": 0, "contact_motivation_recognition": 0, "user_strategy_switching": 0, "reason": f"Evaluation failed: {reason}"},
            "business_rule_capability": {"sop_compliance": 0, "business_info_authenticity": 0, "processing_fairness": 0, "output_consistency": 0, "dialogue_end_recognition": 0, "reason": f"Evaluation failed: {reason}"},
            "ood_issues": {"ood_issue_recognition": 0, "reason": f"Evaluation failed: {reason}"}
        }
        
        return Response(
            content=json.dumps(default_data, ensure_ascii=False),
            success=False,
            metadata={
                "role": "evaluator",
                "evaluation_type": "logic",
                "evaluation_data": default_data,
                "evaluation_success": False,
                "error": reason
            }
        )


@register_component("agent", "compensation")
class CompensationEvaluationAgent(GenericAgent):
    """Compensation Evaluation Agent - responsible for parsing evaluation results"""
    
    def __init__(self, component_type: str, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(component_type, name, config)
    
    def _initialize_agent(self):
        """Initialize compensation evaluation agent"""
        try:
            self.prompt_template = self._load_prompt(self.config["prompt_file"])
            logger.debug(f"Compensation evaluation agent initialized successfully")
        except KeyError as e:
            raise Exception(f"Compensation evaluation agent initialization failed, missing config: {e}")
        except Exception as e:
            raise Exception(f"Compensation evaluation agent initialization failed: {e}")

    def _parse_response(self, llm_response: str) -> Response:
        """Parse LLM compensation evaluation results"""
        try:
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                evaluation_data = json.loads(json_str)
                
                return Response(
                    content=json.dumps(evaluation_data, ensure_ascii=False),
                    success=True,
                    metadata={
                        "role": "evaluator",
                        "evaluation_type": "compensation",
                        "evaluation_data": evaluation_data,
                        "evaluation_success": True
                    }
                )
            else:
                logger.error("Unable to extract JSON-formatted evaluation results from response")
                return self._get_default_response("Unable to extract JSON")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse evaluation JSON: {e}")
            return self._get_default_response(f"JSON parsing failed: {e}")
        except Exception as e:
            logger.error(f"Failed to parse evaluation response: {e}")
            return self._get_default_response(f"Parsing failed: {e}")

    def _get_default_response(self, reason: str) -> Response:
        """Get default compensation evaluation result (all zeros)"""
        default_data = {
            "compensation_strategy_adaptation": {"negotiation_first": 0, "compensation_timing_control": 0, "evidence_sufficiency": 0, "reason": f"Evaluation failed: {reason}"}
        }
        
        return Response(
            content=json.dumps(default_data, ensure_ascii=False),
            success=False,
            metadata={
                "role": "evaluator",
                "evaluation_type": "compensation",
                "evaluation_data": default_data,
                "evaluation_success": False,
                "error": reason
            }
        )
