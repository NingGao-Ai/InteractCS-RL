import json
import logging
import os
from typing import Dict, Any, List, Optional
from session.base import BaseSession
from core.types import ConversationResult, Response, Context
from core.registry import register_component, ComponentRegistry

logger = logging.getLogger(__name__)


@register_component("session", "Evaluation")
class EvaluationSession(BaseSession):
    """Evaluation Session - for evaluating generated dialogue data"""
    
    def __init__(self, component_type: str, name: str, config: Optional[Dict[str, Any]] = None):
        self.conversations = self._load_conversations(config=config)
        config["num_conversations"] = len(self.conversations)
        super().__init__(component_type, name, config)
        
    def _load_conversations(self, config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Load conversation data"""
        input_file = config.get("input_file")
        if not input_file:
            raise Exception("input_file not specified in evaluation config")
        
        conversations = []
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        conversations.append(json.loads(line))
            
            logger.info(f"Successfully loaded {len(conversations)} conversations")
            
            max_conversations = config.get("max_conversations", -1)
            if max_conversations > 0 and max_conversations < len(conversations):
                conversations = conversations[:max_conversations]
                logger.info(f"Limited evaluation conversations to: {max_conversations}")
            
            return conversations
        except Exception as e:
            logger.error(f"Failed to load conversation data: {e}")
            raise
    
    def initialize(self, num_conversation: int, registry: ComponentRegistry):
        """Initialize evaluation session"""
        self.speech_evaluator = None
        self.logic_evaluator = None
        self.compensation_evaluator = None
        
        evaluator_types = self.config.get("evaluator_types", ["speech", "logic", "compensation", "format"])
        self.enable_format_check = "format" in evaluator_types if evaluator_types else False
        
        for eval_type in evaluator_types or []:
            if eval_type == "format":
                continue
            
            try:
                evaluator = registry.get("agent", eval_type)
                if eval_type == "speech":
                    self.speech_evaluator = evaluator
                elif eval_type == "logic":
                    self.logic_evaluator = evaluator
                elif eval_type == "compensation":
                    self.compensation_evaluator = evaluator
            except Exception as e:
                logger.warning(f"Unable to load evaluator {eval_type}: {e}")
        
        self.num_conversationts = num_conversation
        logger.info(f"Evaluation session initialized, will evaluate {self.num_conversationts} conversations")
    
    def start_conversation(self, index: int) -> ConversationResult:
        """Evaluate a single conversation"""
        results = ConversationResult()
        
        try:
            conversation = self.conversations[index]
            conversation_id = conversation.get("conversation_index", index)
            dialogue_history = conversation.get("dialogue", [])
            if not dialogue_history:
                logger.warning(f"Conversation {conversation_id} has no dialogue history")
                return results
            
            evaluation_data = {}
            
            if self.enable_format_check:
                format_check_result = self._calculate_format_check(dialogue_history, {})
                evaluation_data["format_check"] = format_check_result
            
            if self.speech_evaluator:
                try:
                    speech_result = self._evaluate_speech(dialogue_history)
                    evaluation_data["speech_evaluation"] = speech_result.metadata.get("evaluation_data", {})
                    evaluation_data["speech_evaluation_success"] = speech_result.metadata.get("evaluation_success", False)
                    results.add_result(speech_result)
                except Exception as e:
                    logger.error(f"Speech evaluation failed: {e}")
                    evaluation_data["speech_evaluation"] = {"error": str(e)}
                    evaluation_data["speech_evaluation_success"] = False
            
            if self.logic_evaluator:
                try:
                    logic_result = self._evaluate_logic(dialogue_history)
                    evaluation_data["logic_evaluation"] = logic_result.metadata.get("evaluation_data", {})
                    evaluation_data["logic_evaluation_success"] = logic_result.metadata.get("evaluation_success", False)
                    results.add_result(logic_result)
                except Exception as e:
                    logger.error(f"Logic evaluation failed: {e}")
                    evaluation_data["logic_evaluation"] = {"error": str(e)}
                    evaluation_data["logic_evaluation_success"] = False
            
            if self.compensation_evaluator:
                try:
                    has_voucher = self._has_voucher(dialogue_history)
                    if has_voucher:
                        compensation_result = self._evaluate_compensation(dialogue_history)
                        evaluation_data["compensation_evaluation"] = compensation_result.metadata.get("evaluation_data", {})
                        evaluation_data["compensation_evaluation_success"] = compensation_result.metadata.get("evaluation_success", False)
                        results.add_result(compensation_result)
                    else:
                        evaluation_data["compensation_evaluation"] = None
                        evaluation_data["compensation_evaluation_success"] = None
                except Exception as e:
                    logger.error(f"Compensation evaluation failed: {e}")
                    evaluation_data["compensation_evaluation"] = {"error": str(e)}
                    evaluation_data["compensation_evaluation_success"] = False
            
            if not results.results:
                results.add_result(Response(content="", success=True, metadata={}))
            
            if results.results:
                results.results[0].metadata["all_evaluations"] = evaluation_data
                results.results[0].metadata["conversation_id"] = conversation_id
                results.results[0].metadata["original_conversation"] = conversation
            
        except Exception as e:
            logger.error(f"Evaluation of conversation {index} failed: {e}", exc_info=True)
            results.add_result(Response(content="", success=False, metadata={"error": str(e)}))
        
        return results
    
    def _evaluate_speech(self, dialogue_history: List[Dict[str, Any]]) -> Response:
        """Execute speech evaluation"""
        dialogue_text = self._format_dialogue_for_speech(dialogue_history)
        prompt = self.speech_evaluator.prompt_template.replace("{{DIALOGUE}}", dialogue_text)
        context = Context()
        return self.speech_evaluator.generate_response(prompt, context)
    
    def _evaluate_logic(self, dialogue_history: List[Dict[str, Any]]) -> Response:
        """Execute logic evaluation"""
        dialogue_text = self._format_dialogue_for_logic(dialogue_history)
        prompt = self.logic_evaluator.prompt_template.replace("{{DIALOGUE}}", dialogue_text)
        context = Context()
        return self.logic_evaluator.generate_response(prompt, context)
    
    def _evaluate_compensation(self, dialogue_history: List[Dict[str, Any]]) -> Response:
        """Execute compensation evaluation"""
        dialogue_text = self._format_dialogue_for_logic(dialogue_history)
        prompt = self.compensation_evaluator.prompt_template.replace("{{DIALOGUE}}", dialogue_text)
        context = Context()
        return self.compensation_evaluator.generate_response(prompt, context)
    
        
    def _format_dialogue_for_speech(self, dialogue_history: List[Dict[str, Any]]) -> str:
        """Format dialogue for speech evaluation"""
        dialogue_lines = []
        for turn in dialogue_history:
            metadata = turn.get("metadata", {})
            role = metadata.get("role")
            content = turn.get("content", "")
            if role == "user":
                dialogue_lines.append(f"user: {content}")
            elif role == "assistant":
                dialogue_lines.append(f"assistant: {content}")
        return "\n".join(dialogue_lines)
    
    def _format_dialogue_for_logic(self, dialogue_history: List[Dict[str, Any]]) -> str:
        """Format dialogue for logic/compensation evaluation"""
        dialogue_lines = []
        for turn in dialogue_history:
            metadata = turn.get("metadata", {})
            role = metadata.get("role")
            if role == "user":
                content = turn.get("content", "")
                dialogue_lines.append(f"user: {content}")
            elif role == "assistant":
                content = metadata.get("full_response", "")
                dialogue_lines.append(f"assistant: {content}")
        return "\n".join(dialogue_lines)
    
    def _calculate_format_check(self, dialogue_history: List[Dict[str, Any]], evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate format check results"""
        voucher_count = sum(1 for turn in dialogue_history
                          if turn.get("metadata", {}).get("action") == "voucher")
        multiple_vouchers = voucher_count > 1
        total_assistant_turns = sum(1 for turn in dialogue_history
                                   if turn.get("metadata", {}).get("role") == "assistant")
        
        format_errors = []
        format_correct = True
        if multiple_vouchers:
            format_errors.append("multiple_vouchers")
            format_correct = False
        
        return {
            "format_correct": format_correct,
            "is_correct": format_correct,
            "status": "correct" if format_correct else "incorrect",
            "multiple_vouchers": multiple_vouchers,
            "format_errors": format_errors,
            "voucher_count": voucher_count,
            "total_assistant_turns": total_assistant_turns
        }
    
    def _has_voucher(self, dialogue_history: List[Dict[str, Any]]) -> bool:
        """Check if the conversation contains a voucher/compensation"""
        for turn in dialogue_history:
            role = turn.get("role", "unknown")
            agent_type = turn.get("agent_type", role)
            
            if agent_type == "assistant" or role == "assistant":
                metadata = turn.get("metadata", {})
                action = metadata.get("action", "")
                if action == "voucher":
                    return True
        return False
    
    def custom_result(self, conversation_result: ConversationResult, index: int = None) -> Dict[str, Any]:
        """Custom result storage format"""
        if not conversation_result.results:
            return {
                "conversation_index": index,
                "evaluation_status": "failed",
                "error": "No evaluation results"
            }
        
        first_result = conversation_result.results[0]
        evaluation_data = first_result.metadata.get("all_evaluations", {})
        conversation_id = first_result.metadata.get("conversation_id", index)
        original_conversation = first_result.metadata.get("original_conversation", {})
        total_scores = self._calculate_total_scores(evaluation_data)
        
        # Extract user satisfaction
        satisfaction = original_conversation.get("summary", {}).get("satisfaction", {}).get("satisfaction")
        satisfaction_stats = {
            "has_satisfaction": satisfaction is not None,
            "satisfaction": satisfaction
        }
        
        return {
            "conversation_id": conversation_id,
            "evaluation_status": "success" if first_result.success else "failed",
            "evaluations": evaluation_data,
            "scores": total_scores,
            "satisfaction": satisfaction_stats,
            "original_conversation": original_conversation
        }
    
    def _calculate_total_scores(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate total scores for each evaluation type"""
        scores = {}
        
        if "speech_evaluation" in evaluation_data:
            speech_eval = evaluation_data["speech_evaluation"]
            if "error" not in speech_eval:
                speech_score = 0
                for category in ["identity_neutrality", "dialogue_quality", "language_adaptability", "content_quality", "communication_effectiveness", "natural_fluency", "context_adaptability"]:
                    if category in speech_eval and isinstance(speech_eval[category], dict):
                        speech_score += sum(v for k, v in speech_eval[category].items() if k != "reason" and isinstance(v, (int, float)))
                scores["speech_score"] = speech_score
                scores["speech_max_score"] = 28
        
        if "logic_evaluation" in evaluation_data:
            logic_eval = evaluation_data["logic_evaluation"]
            if "error" not in logic_eval:
                logic_score = 0
                for category in ["user_profile_recognition", "business_rule_capability", "ood_issues"]:
                    if category in logic_eval and isinstance(logic_eval[category], dict):
                        logic_score += sum(v for k, v in logic_eval[category].items() if k != "reason" and isinstance(v, (int, float)))
                scores["logic_score"] = logic_score
                scores["logic_max_score"] = 12
        
        if "compensation_evaluation" in evaluation_data and evaluation_data["compensation_evaluation"] is not None:
            comp_eval = evaluation_data["compensation_evaluation"]
            if "error" not in comp_eval and "compensation_strategy_adaptation" in comp_eval:
                comp_data = comp_eval["compensation_strategy_adaptation"]
                if isinstance(comp_data, dict):
                    comp_score = sum(v for k, v in comp_data.items() if k != "reason" and isinstance(v, (int, float)))
                    scores["compensation_score"] = comp_score
                    scores["compensation_max_score"] = 6
        
        total_score = sum(v for k, v in scores.items() if k.endswith("_score"))
        total_max_score = sum(v for k, v in scores.items() if k.endswith("_max_score"))
        scores["total_score"] = total_score
        scores["total_max_score"] = total_max_score
        
        return scores
    
    def post_process(self, result: Dict[str, Any]) -> None:
        """Generate statistics report after evaluation completes"""
        try:
            from utils.evaluation_statistics import EvaluationStatistics
            
            output_file = result.get("output_file")
            if not output_file or not os.path.exists(output_file):
                logger.warning("Evaluation result file does not exist, skipping report generation")
                return
            
            logger.info("Generating evaluation statistics report...")
            statistics = EvaluationStatistics(output_file)
            statistics.print_summary()
            
            report_file = output_file.replace('.jsonl', '_report.json')
            statistics.save_report(report_file)
        except Exception as e:
            logger.error(f"Failed to generate evaluation report: {e}", exc_info=True)
