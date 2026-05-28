# 用户-智能客服模拟交互框架

一个灵活、可扩展的对话模拟和评测框架，支持用户模拟、客服模拟和多维度对话评测。

## 核心功能

- **对话模拟**：基于用户画像和系统信号生成真实的用户-客服对话
- **对话评测**：支持话术、逻辑、赔付、格式等多维度自动化评测
- **灵活配置**：YAML配置驱动，支持命令行参数覆盖
- **并发处理**：支持多线程并发生成和评测对话
- **可扩展架构**：通过注册机制轻松扩展新的Agent和Session

## 快速开始

### 1. 安装依赖

```bash
pip install pyyaml requests
```

### 2. 对话模拟

#### 配置文件

编辑 `config/default.yaml` 或 `config/vllm_customer.yaml`：

```yaml
# 应用配置
app:
  log_level: INFO
  logger_dir: logs/log_results
  log_to_file: true

# Agent配置（支持多个agent）
agent:
  - name: user
    llm:
      client_type: openai
      model: deepseek-chat
      api_url: <your_api_url>
      api_key: <your_api_key>
      kwargs:
        temperature: 1.2
        max_tokens: 8000
    prompt_file: <path_to_user_prompt>
    user_profiles_file: <path_to_user_profiles>
    user_core_need_file: <path_to_core_needs>
  
  - name: customer
    llm:
      client_type: vllm  # 或 openai
      model: <model_path_or_name>
      api_url: http://localhost:8000/v1
      kwargs:
        temperature: 0.7
        max_tokens: 8000
    prompt_file: <path_to_customer_prompt>

# Session配置
session:
  - name: RLSimulation
    num_conversations: 80
    max_workers: 20
    max_turns: 20
    save_batch_size: 20
    system_signals_file: <path_to_system_signals>
    output_dir: outputs/conversations
    user_category_distribution:
      "1": 8
      "2": 24
      "3": 20
      "4": 20
      "5": 8
```

#### 运行模拟

```bash
# 使用默认配置
sh sh/run_framework.sh

# 使用自定义配置
python main.py --config config/vllm_customer.yaml

# 命令行覆盖参数
python main.py --config config/default.yaml \
    --session.RLSimulation.num_conversations=100 \
    --agent.customer.llm.model=qwen-turbo
```

### 3. 对话评测

#### 配置文件

编辑 `config/evaluation.yaml`：

```yaml
# 应用配置
app:
  log_level: INFO
  logger_dir: logs/evaluation_logs
  log_to_file: true

# 评测Agent配置
agent:
  - name: speech      # 话术评测
    llm:
      client_type: openai
      model: deepseek-chat
      api_url: <your_api_url>
      api_key: <your_api_key>
      kwargs:
        temperature: 0.3
        max_tokens: 4000
    prompt_file: <path_to_speech_evaluation_prompt>
  
  - name: logic       # 逻辑评测
    llm:
      client_type: openai
      model: deepseek-chat
      api_url: <your_api_url>
      api_key: <your_api_key>
    prompt_file: <path_to_logic_evaluation_prompt>
  
  - name: compensation  # 赔付评测
    llm:
      client_type: openai
      model: deepseek-chat
      api_url: <your_api_url>
      api_key: <your_api_key>
    prompt_file: <path_to_compensation_evaluation_prompt>

# Session配置
session:
  - name: Evaluation
    input_file: <path_to_conversation_file>
    output_dir: outputs/evaluation
    max_workers: 40
    save_batch_size: 20
    evaluator_types:
      - speech
      - logic
      - compensation
      - format  # 格式检查（内置）
```

#### 运行评测

```bash
# 使用脚本
sh sh/run_evaluation.sh

# 使用命令行
python main.py --config config/evaluation.yaml \
    --session.Evaluation.input_file=outputs/conversations/conversation_80.jsonl \
    --session.Evaluation.output_dir=outputs/evaluation
```

## 项目结构

```
customer_service_simulation_framework/
├── agents/                      # Agent组件
│   ├── base/                    # 基础Agent类
│   ├── UserSimulatorAgent.py    # 用户模拟Agent
│   ├── CustomerServiceAgent.py  # 客服模拟Agent
│   └── EvaluationAgent.py       # 评测Agent
├── session/                     # Session组件
│   ├── base/                    # 基础Session类
│   ├── RLSession.py             # 对话模拟Session
│   └── EvaluationSession.py     # 对话评测Session
├── llm/                         # LLM客户端
│   ├── BaseLLMClient.py
│   ├── OpenAIClient.py
│   └── VLLMClient.py
├── core/                        # 核心模块
│   ├── registry.py              # 组件注册和配置管理
│   └── types.py                 # 数据类型定义
├── utils/                       # 工具模块
│   └── evaluation_statistics.py # 评测统计工具
├── config/                      # 配置文件
│   ├── default.yaml             # 对话模拟配置
│   ├── vllm_customer.yaml       # VLLM配置示例
│   └── evaluation.yaml          # 对话评测配置
├── sh/                          # 启动脚本
│   ├── run_framework.sh         # 对话模拟脚本
│   └── run_evaluation.sh        # 对话评测脚本
└── main.py                      # 应用入口
```

## 配置说明

### 命令行参数覆盖

支持通过命令行参数覆盖配置文件中的任何参数：

```bash
# 格式：--<component_type>.<component_name>.<parameter>=<value>
python main.py --config config/default.yaml \
    --agent.customer.llm.model=qwen-turbo \
    --session.RLSimulation.num_conversations=100 \
    --session.RLSimulation.max_workers=40
```

### Agent配置

```yaml
agent:
  - name: <agent_name>          # Agent名称（user/customer/speech/logic/compensation）
    llm:
      client_type: openai       # LLM客户端类型：openai 或 vllm
      model: <model_name>       # 模型名称或路径
      api_url: <api_url>        # API地址
      api_key: <api_key>        # API密钥（可选）
      max_retries: 3            # 最大重试次数
      kwargs:                   # LLM参数
        temperature: 0.7
        max_tokens: 8000
        timeout: 180
    prompt_file: <path>         # Prompt文件路径
    # Agent特定配置...
```

### Session配置

#### 对话模拟Session

```yaml
session:
  - name: RLSimulation
    num_conversations: 80       # 对话总数
    max_workers: 20             # 并发线程数
    max_turns: 20               # 单个对话最大轮数
    save_batch_size: 20         # 批量保存大小
    system_signals_file: <path> # 系统信号文件
    output_dir: <path>          # 输出目录
    user_category_distribution: # 用户类别分布
      "1": 8
      "2": 24
      "3": 20
      "4": 20
      "5": 8
```

#### 对话评测Session

```yaml
session:
  - name: Evaluation
    input_file: <path>          # 输入对话文件
    output_dir: <path>          # 输出目录
    max_workers: 40             # 并发线程数
    save_batch_size: 20         # 批量保存大小
    evaluator_types:            # 评测类型
      - speech                  # 话术评测
      - logic                   # 逻辑评测
      - compensation            # 赔付评测
      - format                  # 格式检查（内置）
```å

## 输出格式

### 对话模拟输出

```json
{
  "conversation_index": 0,
  "summary": {
    "total_turns": 10,
    "user_turns": 5,
    "assistant_turns": 5,
    "all_success": true,
    "end_reason": "user_ended"
  },
  "user_profile": {...},
  "system_signals": [...],å
  "core_need": {...},
  "dialogue": [
    {
      "turn": 1,
      "role": "用户",
      "agent_type": "user",
      "content": "...",
      "metadata": {...}
    }
  ]
}
```

### 对话评测输出

```json
{
  "conversation_id": 0,
  "evaluation_status": "success",
  "evaluations": {
    "speech_evaluation": {...},
    "logic_evaluation": {...},
    "compensation_evaluation": {...},
    "format_check": {
      "format_correct": true,
      "multiple_vouchers": false,
      "voucher_count": 1,
      "total_assistant_turns": 5
    }
  },
  "scores": {
    "speech_score": 26,
    "speech_max_score": 28,
    "logic_score": 11,
    "logic_max_score": 12,
    "compensation_score": 5,
    "compensation_max_score": 6,
    "total_score": 42,
    "total_max_score": 46
  }
}
```

## 扩展指南

### 添加新的Agent

#### 1. 创建Agent类

在 `agents/` 目录下创建新的Agent文件，例如 `MyCustomAgent.py`：

```python
from agents.base.GenericAgent import GenericAgent
from core.types import Response
from core.registry import register_component

@register_component("agent", "my_custom")
class MyCustomAgent(GenericAgent):
    def _initialize_agent(self):
        """加载必要的资源"""
        # 加载prompt模板
        self.prompt_template = self._load_prompt(self.config["prompt_file"])
        
        # 加载其他数据文件（如果需要）
        if "data_file" in self.config:
            self.data = self._load_data(self.config["data_file"])
    
    def _parse_response(self, llm_response: str) -> Response:
        """解析LLM响应"""
        # 解析LLM输出，提取内容和元数据
        # 示例：假设LLM输出格式为 "content: xxx\nstatus: yyy"
        lines = llm_response.strip().split('\n')
        content = ""
        metadata = {}
        
        for line in lines:
            if line.startswith("content:"):
                content = line.replace("content:", "").strip()
            elif line.startswith("status:"):
                metadata["status"] = line.replace("status:", "").strip()
        
        return Response(
            content=content,
            success=True,
            metadata=metadata
        )
```

#### 2. 配置Agent

在配置文件中添加Agent配置：

```yaml
agent:
  - name: my_custom          # 与 @register_component 中的名称一致
    llm:
      client_type: openai
      model: gpt-4
      api_url: https://api.openai.com/v1
      api_key: your_api_key
      max_retries: 3
      kwargs:
        temperature: 0.7
        max_tokens: 4000
        timeout: 180
    prompt_file: /path/to/my_custom_prompt.txt
    data_file: /path/to/my_data.json  # Agent特定的配置
```

#### 3. 在Session中使用

```python
def initialize(self, num_conversation: int, registry):
    # 获取自定义Agent
    self.my_agent = registry.get("agent", "my_custom")
```

#### 4. 命令行覆盖参数

```bash
python main.py --config config/my_config.yaml \
    --agent.my_custom.llm.model=gpt-4-turbo \
    --agent.my_custom.llm.temperature=0.9
```

### 添加新的Session

#### 1. 创建Session类

在 `session/` 目录下创建新的Session文件，例如 `MyCustomSession.py`：

```python
from session.base import BaseSession
from core.types import ConversationResult, Context
from core.registry import register_component, ComponentRegistry

@register_component("session", "MyCustom")
class MyCustomSession(BaseSession):
    def initialize(self, num_conversation: int, registry: ComponentRegistry):
        """初始化Session"""
        # 获取需要的Agent实例
        self.agent1 = registry.get("agent", "my_custom")
        self.agent2 = registry.get("agent", "another_agent")
        
        # 加载数据文件
        if "data_file" in self.config:
            self.data = self._load_data(self.config["data_file"])
        
        # 初始化其他属性
        self.num_conversationts = num_conversation
    
    def start_conversation(self, index: int) -> ConversationResult:
        """执行单个对话"""
        results = ConversationResult()
        context = Context()
        
        # 对话逻辑
        for turn in range(self.config.get("max_turns", 10)):
            # 1. 构建system_message
            system_message = self._build_system_message(index, turn)
            
            # 2. 调用Agent生成响应
            response = self.agent1.generate_response(system_message, context)
            results.add_result(response)
            
            # 3. 更新context
            context.add_message(response)
            
            # 4. 检查是否结束
            if response.metadata.get("is_end"):
                break
        
        return results
    
    def custom_result(self, conversation_result: ConversationResult, index: int) -> dict:
        """自定义结果格式"""
        return {
            "conversation_id": index,
            "total_turns": len(conversation_result.results),
            "dialogue": [
                {
                    "turn": idx + 1,
                    "content": r.content,
                    "metadata": r.metadata
                }
                for idx, r in enumerate(conversation_result.results)
            ]
        }
    
    def _build_system_message(self, index: int, turn: int) -> str:
        """构建system_message"""
        # 根据业务逻辑构建system_message
        return f"This is turn {turn} of conversation {index}"
    
    def _load_data(self, file_path: str):
        """加载数据文件"""
        import json
        with open(file_path, 'r') as f:
            return json.load(f)
```

#### 2. 配置Session

在配置文件中添加Session配置：

```yaml
session:
  - name: MyCustom          # 与 @register_component 中的名称一致
    num_conversations: 100  # 对话总数
    max_workers: 20         # 并发线程数
    max_turns: 15           # 单个对话最大轮数
    save_batch_size: 20     # 批量保存大小
    output_dir: outputs/my_custom
    data_file: /path/to/session_data.json  # Session特定的配置
```

#### 3. 运行Session

```bash
# 使用配置文件
python main.py --config config/my_config.yaml

# 命令行覆盖参数
python main.py --config config/my_config.yaml \
    --session.MyCustom.num_conversations=200 \
    --session.MyCustom.max_workers=40
```

### 配置文件完整示例

创建一个完整的配置文件 `config/my_custom.yaml`：

```yaml
# 应用配置
app:
  log_level: INFO
  logger_dir: logs/my_custom
  log_to_file: true

# 基础路径
base_path: /path/to/workspace

# Agent配置
agent:
  - name: my_custom
    llm:
      client_type: openai
      model: gpt-4
      api_url: https://api.openai.com/v1
      api_key: your_api_key
      max_retries: 3
      kwargs:
        temperature: 0.7
        max_tokens: 4000
        timeout: 180
    prompt_file: /path/to/prompt.txt
    data_file: /path/to/data.json
  
  - name: another_agent
    llm:
      client_type: vllm
      model: /path/to/model
      api_url: http://localhost:8000/v1
      kwargs:
        temperature: 0.8
        max_tokens: 2000
    prompt_file: /path/to/another_prompt.txt

# Session配置
session:
  - name: MyCustom
    num_conversations: 100
    max_workers: 20
    max_turns: 15
    save_batch_size: 20
    output_dir: outputs/my_custom
    data_file: /path/to/session_data.json
```

### 扩展要点

#### Agent扩展要点

1. **注册名称**：`@register_component("agent", "name")` 中的名称必须与配置文件中的 `name` 一致
2. **配置访问**：通过 `self.config["key"]` 访问配置参数
3. **LLM客户端**：框架自动管理，通过 `self.llm_client` 访问
4. **Prompt加载**：使用 `self._load_prompt(path)` 加载prompt文件

#### Session扩展要点

1. **注册名称**：`@register_component("session", "Name")` 中的名称必须与配置文件中的 `name` 一致
2. **Agent获取**：通过 `registry.get("agent", "name")` 获取Agent实例
3. **配置访问**：通过 `self.config["key"]` 访问配置参数
4. **并发处理**：框架自动处理，只需实现 `start_conversation()` 单个对话逻辑
5. **结果保存**：框架自动保存，通过 `custom_result()` 自定义格式

## 常见问题

### 1. 如何切换不同的模型？

通过命令行参数覆盖：

```bash
python main.py --config config/default.yaml \
    --agent.customer.llm.model=qwen-turbo
```

### 2. 如何调整并发数？

```bash
python main.py --config config/default.yaml \
    --session.RLSimulation.max_workers=40
```

### 3. 如何只运行格式检查？

在 `evaluation.yaml` 中只启用 `format`：

```yaml
evaluator_types:
  - format
```

### 4. 评测结果在哪里？

- 对话文件：`<output_dir>/conversation_<num>_sample_<timestamp>.jsonl`
- 评测文件：`<output_dir>/evaluation_<num>_sample_<timestamp>.jsonl`
- 统计报告：`<output_dir>/evaluation_<num>_sample_<timestamp>_report.json`

## 参考实现

- **对话模拟**：`session/RLSession.py`
- **对话评测**：`session/EvaluationSession.py`
- **用户Agent**：`agents/UserSimulatorAgent.py`
- **客服Agent**：`agents/CustomerServiceAgent.py`
- **评测Agent**：`agents/EvaluationAgent.py`
