from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class Response:
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {"content": self.content, "metadata": self.metadata, "success": self.success}

@dataclass 
class ConversationResult:
    results: List[Response] = field(default_factory=list)
    
    def add_result(self, result: Response) -> None:
        self.results.append(result)


@dataclass
class Context:
    messages: List[Response] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, response: Response) -> None:
        self.messages.append(response)


@dataclass
class Conversation:
    conversation_id: str
    context: Context
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration: float = 0.0
    success: bool = True
    
    