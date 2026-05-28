"""
Evaluation result data structures
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json
import time


@dataclass
class SpeechEvaluationResult:
    """Speech evaluation result"""
    identity_neutrality: Dict[str, int] = field(default_factory=lambda: {
        "identity_claim_avoidance": 0,
        "alternative_expression": 0
    })
    dialogue_quality: Dict[str, int] = field(default_factory=lambda: {
        "content_novelty": 0,
        "problem_focus": 0
    })
    language_adaptability: Dict[str, int] = field(default_factory=lambda: {
        "language_style_matching": 0,
        "technical_complexity_adaptation": 0
    })
    content_quality: Dict[str, int] = field(default_factory=lambda: {
        "content_substantiveness": 0,
        "promise_boundary_control": 0,
        "information_authenticity": 0
    })
    communication_effectiveness: Dict[str, int] = field(default_factory=lambda: {
        "conciseness": 0,
        "sincerity": 0
    })
    natural_fluency: Dict[str, int] = field(default_factory=lambda: {
        "sentence_diversity": 0,
        "transition_naturalness": 0
    })
    context_adaptability: Dict[str, int] = field(default_factory=lambda: {
        "problem_complexity_adaptation": 0,
        "user_knowledge_adaptation": 0
    })
    # Evaluation reasons
    reasons: Dict[str, str] = field(default_factory=lambda: {
        "identity_neutrality": "",
        "dialogue_quality": "",
        "language_adaptability": "",
        "content_quality": "",
        "communication_effectiveness": "",
        "natural_fluency": "",
        "context_adaptability": ""
    })

    @property
    def total_score(self) -> int:
        """Calculate speech evaluation total score"""
        total = 0
        for category in [
            self.identity_neutrality,
            self.dialogue_quality,
            self.language_adaptability,
            self.content_quality,
            self.communication_effectiveness,
            self.natural_fluency,
            self.context_adaptability
        ]:
            total += sum(category.values())
        return total

    @property
    def max_score(self) -> int:
        """Speech evaluation maximum score"""
        return 28  # 14 scoring items, max 2 points each


@dataclass
class LogicEvaluationResult:
    """Logic evaluation result"""
    user_profile_recognition: Dict[str, int] = field(default_factory=lambda: {
        "user_profile_emotion_judgment": 0,
        "contact_motivation_recognition": 0,
        "user_strategy_switching": 0
    })
    business_rule_capability: Dict[str, int] = field(default_factory=lambda: {
        "sop_compliance": 0,
        "business_info_authenticity": 0,
        "processing_fairness": 0,
        "output_consistency": 0,
        "dialogue_end_recognition": 0
    })
    ood_issues: Dict[str, int] = field(default_factory=lambda: {
        "ood_issue_recognition": 0
    })
    # Evaluation reasons
    reasons: Dict[str, str] = field(default_factory=lambda: {
        "user_profile_recognition": "",
        "business_rule_capability": "",
        "ood_issues": ""
    })

    @property
    def total_score(self) -> int:
        """Calculate logic evaluation total score"""
        total = 0
        for category in [
            self.user_profile_recognition,
            self.business_rule_capability,
            self.ood_issues
        ]:
            total += sum(category.values())
        return total

    @property
    def max_score(self) -> int:
        """Logic evaluation maximum score"""
        return 12  # User profile recognition 6 pts + Business rule capability 5 pts + OOD issues 1 pt


@dataclass
class CompensationEvaluationResult:
    """Compensation evaluation result"""
    compensation_strategy_adaptation: Dict[str, int] = field(default_factory=lambda: {
        "negotiation_first": 0,
        "compensation_timing_control": 0,
        "evidence_sufficiency": 0
    })
    # Evaluation reason
    reason: str = ""

    @property
    def total_score(self) -> int:
        """Calculate compensation evaluation total score"""
        return sum(self.compensation_strategy_adaptation.values())

    @property
    def max_score(self) -> int:
        """Compensation evaluation maximum score"""
        return 6  # Negotiation first 2 pts + Compensation timing control 2 pts + Evidence sufficiency 2 pts


@dataclass
class FormatEvaluationResult:
    """Format evaluation result"""
    format_correct: bool = True
    format_errors: List[str] = field(default_factory=list)
    voucher_count: int = 0
    multiple_vouchers: bool = False
    total_assistant_turns: int = 0

    @property
    def has_format_errors(self) -> bool:
        """Whether there are format errors"""
        return len(self.format_errors) > 0


@dataclass
class EvaluationResult:
    """Complete evaluation result"""
    conversation_id: str
    speech_evaluation: SpeechEvaluationResult
    logic_evaluation: LogicEvaluationResult
    compensation_evaluation: Optional[CompensationEvaluationResult] = None
    format_evaluation: FormatEvaluationResult = field(default_factory=FormatEvaluationResult)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_voucher(self) -> bool:
        """Check if the conversation contains a voucher"""
        for turn in self.conversation_history:
            if turn.get('role') == 'assistant' and '<action>voucher</action>' in turn.get('content', ''):
                return True
        return False

    @property
    def has_ood_issue(self) -> bool:
        """Check if OOD issues are involved"""
        ood_score = self.logic_evaluation.ood_issues.get('ood_issue_recognition', 0)
        return ood_score != -1  # Not involved in OOD is -1, involved is 0 or 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        if self.compensation_evaluation:
            result_dict = {
                'conversation_id': self.conversation_id,
                'speech_evaluation': {
                    'identity_neutrality': self.speech_evaluation.identity_neutrality,
                    'dialogue_quality': self.speech_evaluation.dialogue_quality,
                    'language_adaptability': self.speech_evaluation.language_adaptability,
                    'content_quality': self.speech_evaluation.content_quality,
                    'communication_effectiveness': self.speech_evaluation.communication_effectiveness,
                    'natural_fluency': self.speech_evaluation.natural_fluency,
                    'context_adaptability': self.speech_evaluation.context_adaptability,
                    'reasons': self.speech_evaluation.reasons,
                    'total_score': self.speech_evaluation.total_score,
                    'max_score': self.speech_evaluation.max_score
                },
                'logic_evaluation': {
                    'user_profile_recognition': self.logic_evaluation.user_profile_recognition,
                    'business_rule_capability': self.logic_evaluation.business_rule_capability,
                    'ood_issues': self.logic_evaluation.ood_issues,
                    'reasons': self.logic_evaluation.reasons,
                    'total_score': self.logic_evaluation.total_score,
                    'max_score': self.logic_evaluation.max_score
                },
                'compensation_evaluation' : {
                    'compensation_strategy_adaptation': self.compensation_evaluation.compensation_strategy_adaptation,
                    'reason': self.compensation_evaluation.reason,
                    'total_score': self.compensation_evaluation.total_score,
                    'max_score': self.compensation_evaluation.max_score
                },
                'format_evaluation': {
                    'format_correct': self.format_evaluation.format_correct,
                    'format_errors': self.format_evaluation.format_errors,
                    'voucher_count': self.format_evaluation.voucher_count,
                    'multiple_vouchers': self.format_evaluation.multiple_vouchers,
                    'total_assistant_turns': self.format_evaluation.total_assistant_turns
                },
                'conversation_history': self.conversation_history,
                'timestamp': self.timestamp,
                'metadata': self.metadata
            }
        else:
            result_dict = {
                'conversation_id': self.conversation_id,
                'speech_evaluation': {
                    'identity_neutrality': self.speech_evaluation.identity_neutrality,
                    'dialogue_quality': self.speech_evaluation.dialogue_quality,
                    'language_adaptability': self.speech_evaluation.language_adaptability,
                    'content_quality': self.speech_evaluation.content_quality,
                    'communication_effectiveness': self.speech_evaluation.communication_effectiveness,
                    'natural_fluency': self.speech_evaluation.natural_fluency,
                    'context_adaptability': self.speech_evaluation.context_adaptability,
                    'reasons': self.speech_evaluation.reasons,
                    'total_score': self.speech_evaluation.total_score,
                    'max_score': self.speech_evaluation.max_score
                },
                'logic_evaluation': {
                    'user_profile_recognition': self.logic_evaluation.user_profile_recognition,
                    'business_rule_capability': self.logic_evaluation.business_rule_capability,
                    'ood_issues': self.logic_evaluation.ood_issues,
                    'reasons': self.logic_evaluation.reasons,
                    'total_score': self.logic_evaluation.total_score,
                    'max_score': self.logic_evaluation.max_score
                },
                'format_evaluation': {
                    'format_correct': self.format_evaluation.format_correct,
                    'format_errors': self.format_evaluation.format_errors,
                    'voucher_count': self.format_evaluation.voucher_count,
                    'multiple_vouchers': self.format_evaluation.multiple_vouchers,
                    'total_assistant_turns': self.format_evaluation.total_assistant_turns
                },
                'conversation_history': self.conversation_history,
                'timestamp': self.timestamp,
                'metadata': self.metadata
            }

        return result_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationResult':
        """Create instance from dictionary"""
        speech_data = data.get('speech_evaluation', {})
        logic_data = data.get('logic_evaluation', {})
        compensation_data = data.get('compensation_evaluation', {})
        format_data = data.get('format_evaluation', {})

        speech_eval = SpeechEvaluationResult(
            identity_neutrality=speech_data.get('identity_neutrality', {}),
            dialogue_quality=speech_data.get('dialogue_quality', {}),
            language_adaptability=speech_data.get('language_adaptability', {}),
            content_quality=speech_data.get('content_quality', {}),
            communication_effectiveness=speech_data.get('communication_effectiveness', {}),
            natural_fluency=speech_data.get('natural_fluency', {}),
            context_adaptability=speech_data.get('context_adaptability', {}),
            reasons=speech_data.get('reasons', {})
        )

        logic_eval = LogicEvaluationResult(
            user_profile_recognition=logic_data.get('user_profile_recognition', {}),
            business_rule_capability=logic_data.get('business_rule_capability', {}),
            ood_issues=logic_data.get('ood_issues', {}),
            reasons=logic_data.get('reasons', {})
        )

        # If compensation evaluation data exists, create compensation evaluation result
        compensation_eval = None
        if compensation_data:
            compensation_eval = CompensationEvaluationResult(
                compensation_strategy_adaptation=compensation_data.get('compensation_strategy_adaptation', {}),
                reason=compensation_data.get('reason', '')
            )

        format_eval = FormatEvaluationResult(
            format_correct=format_data.get('format_correct', True),
            format_errors=format_data.get('format_errors', []),
            voucher_count=format_data.get('voucher_count', 0),
            multiple_vouchers=format_data.get('multiple_vouchers', False),
            total_assistant_turns=format_data.get('total_assistant_turns', 0)
        )

        return cls(
            conversation_id=data['conversation_id'],
            speech_evaluation=speech_eval,
            logic_evaluation=logic_eval,
            compensation_evaluation=compensation_eval,
            format_evaluation=format_eval,
            conversation_history=data.get('conversation_history', []),
            timestamp=data.get('timestamp', time.time()),
            metadata=data.get('metadata', {})
        )
