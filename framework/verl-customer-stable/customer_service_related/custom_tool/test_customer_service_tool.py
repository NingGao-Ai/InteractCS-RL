#!/usr/bin/env python3
"""
Test script for CustomerServiceTool
"""

import asyncio
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from verl.tools.schemas import OpenAIFunctionToolSchema
from customer_service_tool import CustomerServiceTool


async def test_customer_service_tool():
    """Test the CustomerServiceTool class."""

    # Create tool schema
    tool_schema = OpenAIFunctionToolSchema.model_validate({
        "type": "function",
        "function": {
            "name": "customer_service",
            "description": "A tool for customer service actions like voucher distribution or chat handling",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "The action to perform: 'voucher' for voucher distribution, 'chat' for chat handling",
                        "enum": ["voucher", "chat"]
                    },
                },
                "required": ["action"],
            },
        }
    })

    # Create tool instance
    config = {"type": "native"}
    tool = CustomerServiceTool(config, tool_schema)

    print("=" * 60)
    print("Testing CustomerServiceTool")
    print("=" * 60)

    # Test 1: Create instance
    print("\n1. Creating tool instance...")
    instance_id = await tool.create()
    print(f"   Created instance ID: {instance_id}")

    # Test 2: Execute voucher action
    print("\n2. Testing voucher action...")
    response, reward, metrics = await tool.execute(
        instance_id,
        {"action": "voucher"}
    )
    print(f"   Response: {response}")
    print(f"   Step reward: {reward}")
    print(f"   Metrics: {metrics}")

    # Test 3: Calculate reward
    print("\n3. Calculating final reward...")
    final_reward = await tool.calc_reward(instance_id)
    print(f"   Final reward: {final_reward}")

    # Test 4: Execute chat action (new instance)
    print("\n4. Testing chat action with new instance...")
    instance_id2 = await tool.create()
    response2, reward2, metrics2 = await tool.execute(
        instance_id2,
        {"action": "chat"}
    )
    print(f"   Response: {response2}")
    print(f"   Step reward: {reward2}")
    print(f"   Metrics: {metrics2}")

    # Test 5: Invalid action
    print("\n5. Testing invalid action...")
    response3, reward3, metrics3 = await tool.execute(
        instance_id,
        {"action": "invalid"}
    )
    print(f"   Response: {response3}")
    print(f"   Step reward: {reward3}")
    print(f"   Metrics: {metrics3}")

    # Test 6: Release instances
    print("\n6. Releasing tool instances...")
    await tool.release(instance_id)
    await tool.release(instance_id2)
    print("   Instances released successfully")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_customer_service_tool())
