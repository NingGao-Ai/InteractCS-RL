# Customer Service Tool

A custom tool for handling customer service actions in VERL framework.

## Overview

This tool is triggered when the LLM outputs:
```xml
<tool_call>
{"name": "customer_service", "arguments": {"action": "voucher"}}
</tool_call>
```

or:
```xml
<tool_call>
{"name": "customer_service", "arguments": {"action": "chat"}}
</tool_call>
```

The tool will return:
- `"voucher success"` for voucher action
- `"chat success"` for chat action

## Files

1. **`customer_service_tool.py`** - The main tool implementation
2. **`customer_service_tool_config.yaml`** - Tool configuration file
3. **`test_customer_service_tool.py`** - Test script

## Usage

### 1. Integration with VERL

To use this tool in your VERL configuration:

```yaml
# In your main config file
multi_turn:
  enable: true
  tool_config_path: "customer_service_related/custom_tool/customer_service_tool_config.yaml"
  # ... other configs
```

### 2. Tool Schema

The tool has the following schema:
- **Name**: `customer_service`
- **Description**: "A tool for customer service actions like voucher distribution or chat handling"
- **Parameters**:
  - `action` (required): The action to perform
    - Type: string
    - Enum: ["voucher", "chat"]
    - Description: "The action to perform: 'voucher' for voucher distribution, 'chat' for chat handling"

### 3. LLM Output Format

The LLM should output tool calls in the following format:

```xml
<think>Reasoning about user needs...</think>
<tool_call>
{"name": "customer_service", "arguments": {"action": "voucher"}}
</tool_call>
<response>I'll help you with that voucher.</response>
```

### 4. Tool Behavior

- **Voucher action**: Returns `"voucher success"`
- **Chat action**: Returns `"chat success"`
- **Invalid action**: Returns error message with negative reward
- **Reward**: Gives 0.1 reward for successful tool execution

### 5. Testing

Run the test script to verify the tool works:

```bash
cd framework/verl-customer-stable/customer_service_related/custom_tool
python test_customer_service_tool.py
```

## Implementation Details

### Tool Class Structure

The `CustomerServiceTool` class extends `BaseTool` and implements:
- `create()`: Creates a new tool instance
- `execute()`: Executes the tool with given parameters
- `calc_reward()`: Calculates reward for tool usage
- `release()`: Releases tool instance resources

### Instance Management

The tool maintains an `_instance_dict` to track:
- Action type (voucher/chat)
- Execution status
- Reward history

### Error Handling

- Invalid actions return error messages
- JSON parsing errors are handled gracefully
- Instance management prevents memory leaks

## Integration with Existing Code

To integrate with your existing interaction code, update the action extraction logic:

```python
# Old format: <action>voucher</action>
# New format: <tool_call>{"name": "customer_service", "arguments": {"action": "voucher"}}</tool_call>

import re
import json

def extract_action_from_tool_call(message_content):
    """Extract action from new tool_call format."""
    tool_call_match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', message_content, re.DOTALL)
    if tool_call_match:
        try:
            tool_data = json.loads(tool_call_match.group(1).strip())
            if tool_data.get("name") == "customer_service":
                return tool_data.get("arguments", {}).get("action")
        except json.JSONDecodeError:
            pass
    return None
